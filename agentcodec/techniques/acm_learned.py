"""
ACM with a *learned* router (`acm_learned`).

Replaces the hand-coded difficulty-range MCS table with a multinomial
logistic regression that maps task features to a technique label. Everything
else — pilot-probe difficulty estimation, dispatch to the chosen technique,
logging — is the same as :class:`~agentcodec.techniques.acm.ACMRouter`.

Design rationale
----------------
A reviewer's objection to the hand-coded table is that bin edges and
per-bin winners were tuned on the evaluation tasks. The learned router
addresses this concern directly:

- The fitting objective (argmax-quality technique per task) is explicit,
  reproducible, and has zero manual degrees of freedom beyond the L2
  regularisation strength.
- The router's expected deployment quality is reported as the out-of-fold
  K-fold CV mean from the training script. That number can be compared
  against the hand-coded bins' realised quality without a calibration–
  evaluation leak.
- The decomposition in the paper's ACM oracle-gap analysis
  (info / gen / policy / realisation) still applies; the learned router
  by construction minimises the `policy` term on the chosen feature set.

Interface contract
------------------
A :class:`ACMLearnedRouter` is a drop-in replacement for
:class:`ACMRouter`: same constructor channels/scorer/critic args and same
``run(task)`` return type. The routing table is replaced by a
``RouterWeights`` instance loaded from the JSON produced by the upstream
ACM router trainer (not shipped in the open-source release).

Per-technique dispatch params (rounds, branches, code rate) are *not*
learned; they default to the standalone-technique cache's settings and
can be overridden via ``dispatch_defaults``.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

from ..channel import AgentChannel, QualityScorer
from ..models import CombiningStrategy, HARQMode, ReliabilityRun, TaskItem
from .baselines import SelfConsistencyBaseline, SelfRefineBaseline
from .diversity import DiversityEnsemble, DiversityMRCDiscreteN, SelectionCombiningN
from .fec import FECService
from .fountain import FountainDecoder
from .harq import HARQService
from .turbo import TurboDecoder

logger = logging.getLogger(__name__)


# Per-technique dispatch parameters used when the learned router picks a
# technique. These match the standalone-technique cache's settings so that
# acm_learned's realised quality is directly comparable with the standalone
# harq_ir / turbo / fountain / etc. caches in the same experiment.
DEFAULT_DISPATCH_PARAMS: dict[str, dict[str, Any]] = {
    "baseline":                  {},
    "harq_ir":                   {"max_rounds": 5},
    "harq_cc":                   {"max_rounds": 5},
    "turbo":                     {"max_iterations": 5},
    "fountain":                  {"num_branches": 2},
    "diversity_sc":              {"num_branches": 2},
    "diversity_mrc":             {"num_branches": 2},
    "diversity_egc":             {"num_branches": 2},
    "fec_0.75":                  {"code_rate": 0.75},
    "fec_0.50":                  {"code_rate": 0.50},
    "fec_0.33":                  {"code_rate": 0.33},
    "fec_0.25":                  {"code_rate": 0.25},
    "diversity_sc_N":            {"num_samples": 5},
    "diversity_mrc_discrete_N":  {"num_samples": 5},
    "self_consistency":          {"num_samples": 5},
    "self_refine":               {"max_rounds": 3},
}


# Content-feature computation must stay in lockstep with the upstream
# ACM router trainer (CONTENT_FEATURE_KEYS, content_features), which is
# not shipped in the open-source release.
_CONTENT_FEATURE_KEYS = (
    "log_word_count",
    "has_code",
    "has_math",
    "has_numbers",
    "has_question",
    "log_sentence_count",
    "avg_word_len",
)
_CODE_HINTS = (" def ", " class ", "function ", "import ", "return ", "->",
               "()", "; ", "{}")
_MATH_HINTS = (" integral", " derivative", " equation", " prove", "theorem",
               " lemma", " matrix", " vector", " probability", " variance")
_MATH_CHARS = "$=^∫∑∏√≤≥≠"


def _content_feature_dict(prompt: str) -> dict[str, float]:
    p = prompt or ""
    p_lower = p.lower()
    words = p.split()
    n_words = max(1, len(words))
    n_sentences = max(1, p.count(".") + p.count("?") + p.count("!"))
    avg_word_len = sum(len(w) for w in words) / n_words
    return {
        "log_word_count": math.log1p(n_words) / 10.0,
        "has_code": float("```" in p or any(kw in p_lower for kw in _CODE_HINTS)),
        "has_math": float(any(c in p for c in _MATH_CHARS)
                          or any(kw in p_lower for kw in _MATH_HINTS)),
        "has_numbers": float(any(ch.isdigit() for ch in p)),
        "has_question": float("?" in p),
        "log_sentence_count": math.log1p(n_sentences) / 5.0,
        "avg_word_len": min(avg_word_len, 15.0) / 10.0,
    }


@dataclass
class RouterWeights:
    """Deserialised router — matches the JSON schema from the upstream
    ACM router trainer (not shipped in the open-source release). Backed
    by either a multinomial logit or per-technique ridge regression; the
    forward pass (X @ W argmax) is identical, so callers do not need to
    branch on the training target."""
    version: int
    categories: list[str]
    technique_classes: list[str]
    weights: list[list[float]]           # shape (n_features, n_classes)
    has_bias: bool
    has_difficulty: bool
    has_content_features: bool
    feature_order: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | pathlib.Path) -> RouterWeights:
        p = pathlib.Path(path)
        blob = json.loads(p.read_text())
        spec = blob.get("feature_spec", {})
        cats = list(spec.get("categories", []))
        has_bias = bool(spec.get("has_bias", True))
        has_diff = bool(spec.get("has_difficulty", True))
        has_content = bool(spec.get("has_content_features", False))
        # Backward-compat: derive feature_order if the JSON predates the field.
        fo = list(spec.get("feature_order") or
                  (["bias"] if has_bias else [])
                  + (["difficulty"] if has_diff else [])
                  + [f"cat_{c}" for c in cats]
                  + (list(_CONTENT_FEATURE_KEYS) if has_content else []))
        rw = cls(
            version=int(blob.get("version", 1)),
            categories=cats,
            technique_classes=list(blob["technique_classes"]),
            weights=list(blob["weights"]),
            has_bias=has_bias,
            has_difficulty=has_diff,
            has_content_features=has_content,
            feature_order=fo,
            metadata={k: v for k, v in blob.items()
                      if k not in {"weights", "feature_spec", "technique_classes"}},
        )
        rw._validate()
        return rw

    def _validate(self) -> None:
        n_features_got = len(self.weights)
        if n_features_got != len(self.feature_order):
            raise ValueError(
                f"Router weights have {n_features_got} feature rows but "
                f"feature_order has {len(self.feature_order)} entries"
            )
        n_classes_from_rows = {len(row) for row in self.weights}
        if len(n_classes_from_rows) != 1:
            raise ValueError(f"Ragged weights matrix: row widths {n_classes_from_rows}")
        (n_classes,) = n_classes_from_rows
        if n_classes != len(self.technique_classes):
            raise ValueError(
                f"Weights have {n_classes} columns but {len(self.technique_classes)} "
                f"technique_classes were listed"
            )

    def features_for(
        self, category: str, difficulty: float, prompt: str | None = None,
    ) -> list[float]:
        content = _content_feature_dict(prompt or "") if self.has_content_features else {}
        feats: list[float] = []
        for name in self.feature_order:
            if name == "bias":
                feats.append(1.0)
            elif name == "difficulty":
                feats.append(float(difficulty))
            elif name.startswith("cat_"):
                feats.append(1.0 if category == name[4:] else 0.0)
            elif name in content:
                feats.append(content[name])
            else:
                # Unknown feature: zero-fill so old weights with stale names
                # still produce deterministic output instead of crashing.
                feats.append(0.0)
        return feats

    def predict_logits(
        self, category: str, difficulty: float, prompt: str | None = None,
    ) -> list[float]:
        x = self.features_for(category, difficulty, prompt=prompt)
        n_features = len(x)
        n_classes = len(self.technique_classes)
        logits = [0.0] * n_classes
        for i in range(n_features):
            row = self.weights[i]
            xi = x[i]
            for j in range(n_classes):
                logits[j] += xi * row[j]
        return logits

    def predict_proba(
        self, category: str, difficulty: float, prompt: str | None = None,
    ) -> list[float]:
        logits = self.predict_logits(category, difficulty, prompt=prompt)
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def predict(
        self, category: str, difficulty: float, prompt: str | None = None,
    ) -> tuple[str, float]:
        logits = self.predict_logits(category, difficulty, prompt=prompt)
        best_idx = max(range(len(logits)), key=logits.__getitem__)
        probs = self.predict_proba(category, difficulty, prompt=prompt)
        return self.technique_classes[best_idx], probs[best_idx]


def load_router(path: str | pathlib.Path) -> RouterWeights:
    """Load a linear/ridge router weights JSON.

    SemKNN caches (``router_type: semknn_cost_aware``) are *not* loadable
    here in the public release — that path requires the proprietary
    training data. Use :class:`agentcodec.routing.RemoteSemKNNRouter`
    against a hosted backend instead.
    """
    p = pathlib.Path(path)
    blob = json.loads(p.read_text())
    rt = blob.get("router_type", "linear")
    if rt == "semknn_cost_aware":
        raise ValueError(
            f"Cannot load SemKNN cache {str(p)!r} locally: the SemKNN method "
            f"requires the proprietary training data, which is not "
            f"distributed with this release. Configure a "
            f"`RemoteSemKNNRouter` (or set `router.type: semknn` with a "
            f"`server_url` in your YAML) to use SemKNN routing via a hosted "
            f"backend. See COMMERCIAL.md for on-premise licensing options."
        )
    return RouterWeights.from_path(p)


class ACMLearnedRouter:
    """
    ACM with a learned (logit) routing policy.

    The pilot-probe difficulty estimator is identical to
    :class:`~agentcodec.techniques.acm.ACMRouter`: a short generation with
    logprobs enabled; difficulty = 1 - exp(mean_logprob). The mapping from
    (category, difficulty) to technique is performed by the loaded
    ``RouterWeights`` instead of by hand-coded bins.
    """

    def __init__(
        self,
        channels: dict[str, AgentChannel],
        scorer: QualityScorer,
        router_weights: RouterWeights | str | pathlib.Path,
        dispatch_defaults: dict[str, dict[str, Any]] | None = None,
        difficulty_estimator: AgentChannel | None = None,
        critic_channel: AgentChannel | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        if isinstance(router_weights, RouterWeights):
            self.router = router_weights
        else:
            self.router = load_router(router_weights)
        # Merge user-supplied overrides onto the module defaults.
        merged = {k: dict(v) for k, v in DEFAULT_DISPATCH_PARAMS.items()}
        for k, v in (dispatch_defaults or {}).items():
            merged.setdefault(k, {}).update(v)
        self.dispatch_defaults = merged
        self.difficulty_estimator = difficulty_estimator or next(iter(channels.values()))
        self.critic_channel = critic_channel

    # ---- public entrypoint --------------------------------------------------

    def run(self, task: TaskItem) -> ReliabilityRun:
        difficulty, diff_output = self._estimate_difficulty(task)
        cat = task.category.value if hasattr(task.category, "value") else str(task.category)
        predicted_tech, prob = self.router.predict(cat, difficulty, prompt=task.prompt)

        logger.info(
            f"ACMLearned routing: category={cat} difficulty={difficulty:.2f} "
            f"→ {predicted_tech} (p={prob:.2f})"
        )

        run = self._execute(task, predicted_tech)
        run.overhead_outputs.append(diff_output)
        run.config["estimated_difficulty"] = difficulty
        run.config["router"] = "acm_learned"
        run.config["routed_technique"] = predicted_tech
        run.config["router_prob"] = prob
        run.config["routing_category"] = cat
        if diff_output.mean_logprob is not None:
            run.config["difficulty_source"] = "pilot_logprob"
            run.config["difficulty_logprob"] = diff_output.mean_logprob
        else:
            run.config["difficulty_source"] = "self_rating"
        run.config["router_weights"] = self.router.metadata.get("trained_on", "unknown")
        run.config["router_cv_mean_q"] = self.router.metadata.get("cv_mean_q")
        run.compute_metrics()
        return run

    # ---- internals ----------------------------------------------------------

    def _estimate_difficulty(self, task: TaskItem):
        """Pilot-probe difficulty estimator — copied verbatim from ACMRouter
        so the pilot-probe cost model is identical between the hand-coded
        and learned routers."""
        from .soft import _has_logprobs, _logprob_to_confidence

        probe_prompt = (
            f"Give a brief, direct answer to the following (2-3 sentences max):\n\n"
            f"{task.prompt}"
        )
        try:
            result = self.difficulty_estimator.transmit(
                probe_prompt, temperature=0.3, request_logprobs=True,
            )
        except Exception:
            result = self.difficulty_estimator.transmit(
                probe_prompt, temperature=0.3,
            )
        if _has_logprobs(result):
            confidence = _logprob_to_confidence(result.mean_logprob)
            difficulty = max(0.0, min(1.0, 1.0 - confidence))
            return difficulty, result

        # Fallback: self-rating — kept for backends without logprobs.
        prompt = (
            f"Rate the difficulty of this task on a scale from 0.0 to 1.0, "
            f"where 0.0 is trivial and 1.0 is extremely challenging.\n\n"
            f"Task category: {task.category.value}\n"
            f"Task: {task.prompt}\n\n"
            f'Respond with ONLY a JSON object: {{"difficulty": <float 0-1>, "reasoning": "<brief>"}}'
        )
        result = self.difficulty_estimator.transmit(prompt, temperature=0.1)
        try:
            text = QualityScorer._strip_thinking(result.text.strip())
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            return max(0.0, min(1.0, float(parsed.get("difficulty", 0.5)))), result
        except Exception:
            return 0.5, result

    def _execute(self, task: TaskItem, technique: str) -> ReliabilityRun:
        """Dispatch to the chosen technique with the module-default params."""
        primary_channel = next(iter(self.channels.values()))
        params = self.dispatch_defaults.get(technique, {})

        if technique == "baseline":
            return self._run_baseline(task, primary_channel)

        if technique.startswith("fec_"):
            rate = float(params.get("code_rate", technique.split("_", 1)[1]))
            svc = FECService(primary_channel, self.scorer, code_rate=rate)
            run = svc.run(task)
            run.technique = f"acm_learned_{technique}"
            return run

        if technique == "harq_ir":
            svc = HARQService(
                primary_channel, self.scorer, mode=HARQMode.IR,
                max_rounds=int(params.get("max_rounds", 5)),
                critic_channel=self.critic_channel,
            )
            run = svc.run(task)
            run.technique = "acm_learned_harq_ir"
            return run

        if technique == "harq_cc":
            svc = HARQService(
                primary_channel, self.scorer, mode=HARQMode.CC,
                max_rounds=int(params.get("max_rounds", 5)),
                critic_channel=self.critic_channel,
            )
            run = svc.run(task)
            run.technique = "acm_learned_harq_cc"
            return run

        if technique == "turbo":
            svc = TurboDecoder(
                generator=primary_channel,
                critic=self.critic_channel or primary_channel,
                scorer=self.scorer,
                max_iterations=int(params.get("max_iterations", 5)),
            )
            run = svc.run(task)
            run.technique = "acm_learned_turbo"
            return run

        if technique == "fountain":
            available = list(self.channels.values())
            n = max(int(params.get("num_branches", 2)), 1)
            channels_list = available[:n] if len(available) >= n else [primary_channel] * n
            svc = FountainDecoder(channels=channels_list, scorer=self.scorer)
            run = svc.run(task)
            run.technique = "acm_learned_fountain"
            return run

        if technique in {"diversity_sc", "diversity_mrc", "diversity_egc"}:
            available = list(self.channels.values())
            n = max(int(params.get("num_branches", 2)), 1)
            channels_list = available[:n] if len(available) >= n else [primary_channel] * n
            strategy = {
                "diversity_sc":  CombiningStrategy.SC,
                "diversity_mrc": CombiningStrategy.MRC,
                "diversity_egc": CombiningStrategy.EGC,
            }[technique]
            svc = DiversityEnsemble(channels_list, self.scorer, combining=strategy)
            run = svc.run(task)
            run.technique = f"acm_learned_{technique}"
            return run

        # Wider-pool / multi-model diversity variants.
        if technique == "diversity_sc_N":
            channels_list = list(self.channels.values()) or [primary_channel]
            svc = SelectionCombiningN(
                channels=channels_list, scorer=self.scorer,
                num_samples=int(params.get("num_samples", 5)),
            )
            run = svc.run(task)
            run.technique = "acm_learned_diversity_sc_N"
            return run

        if technique == "diversity_mrc_discrete_N":
            channels_list = list(self.channels.values()) or [primary_channel]
            svc = DiversityMRCDiscreteN(
                channels=channels_list, scorer=self.scorer,
                num_samples=int(params.get("num_samples", 5)),
                voter=primary_channel,
            )
            run = svc.run(task)
            run.technique = "acm_learned_diversity_mrc_discrete_N"
            return run

        # Single-model prior-method reproductions (matched-budget).
        if technique == "self_consistency":
            channels_list = list(self.channels.values()) or [primary_channel]
            svc = SelfConsistencyBaseline(
                channels=channels_list, scorer=self.scorer,
                num_samples=int(params.get("num_samples", 5)),
                voter=primary_channel,
            )
            run = svc.run(task)
            run.technique = "acm_learned_self_consistency"
            return run

        if technique == "self_refine":
            svc = SelfRefineBaseline(
                channel=primary_channel, scorer=self.scorer,
                max_rounds=int(params.get("max_rounds", 3)),
            )
            run = svc.run(task)
            run.technique = "acm_learned_self_refine"
            return run

        # Unknown technique → baseline + warning (should never happen if the
        # training candidate set matches the dispatcher's vocabulary).
        logger.warning(
            f"ACMLearnedRouter: predicted technique {technique!r} has no dispatch — "
            f"falling back to baseline."
        )
        return self._run_baseline(task, primary_channel)

    def _run_baseline(self, task: TaskItem, channel: AgentChannel) -> ReliabilityRun:
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="acm_learned_baseline",
            config={"model": channel.model},
        )
        out = channel.transmit(task.request)
        out.quality_score = self.scorer.score(
            task.prompt, out.text, reference=task.reference, task=task,
        )
        run.individual_outputs = [out]
        run.combined_output = out.text
        run.final_quality = out.quality_score
        run.compute_metrics()
        return run
