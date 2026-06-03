"""
Technique 6: Adaptive Coding & Modulation (ACM)

Communication analog: ACM adapts the modulation scheme and code rate based
on measured channel SNR. Good channel → high rate (fast, less redundancy).
Bad channel → low rate (slow, more protection). This maximizes throughput
for the given channel conditions.

Agent analog: Route tasks to appropriate models and redundancy levels based
on estimated task difficulty. Simple tasks → cheap/fast model, no redundancy.
Hard tasks → powerful model + full redundancy pipeline.

This is the "meta-technique" that orchestrates the others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..channel import AgentChannel, QualityScorer
from ..models import AgentOutput, CombiningStrategy, HARQMode, ReliabilityRun, TaskItem
from .diversity import DiversityEnsemble
from .fec import FECService
from .fountain import FountainDecoder
from .harq import HARQService
from .turbo import TurboDecoder

logger = logging.getLogger(__name__)


@dataclass
class ACMProfile:
    """A modulation+coding scheme for a given difficulty level."""
    name: str
    difficulty_range: tuple[float, float]  # (min, max) estimated difficulty
    model: str
    technique: str             # "uncoded", "fec", "harq_ir", "diversity_mrc"
    code_rate: float = 1.0     # for FEC
    num_branches: int = 1      # for diversity
    max_rounds: int = 1        # for HARQ/turbo
    estimated_cost_multiplier: float = 1.0


# Default ACM profiles (like MCS table in LTE/5G).
# Calibrated from the per-technique quality profile we observe at the paper's
# primary operating points: HARQ-IR dominates the easy-to-medium regime, turbo
# wins the hard regime when the refinement map is contractive, and diversity
# combining provides the safety net at the extreme tail. If the generator is
# sub-threshold (e.g. 3B models) the routing table should be overridden in
# config to avoid turbo — copy configs/lib/routed_acm_table.yaml and tune
# the `table:` block.
DEFAULT_ACM_TABLE: list[ACMProfile] = [
    ACMProfile(
        name="MCS-0: Fast/Cheap",
        difficulty_range=(0.0, 0.3),
        model="gpt-4o-mini",
        technique="harq_ir",
        max_rounds=2,
        estimated_cost_multiplier=2.0,
    ),
    ACMProfile(
        name="MCS-1: Moderate",
        difficulty_range=(0.3, 0.5),
        model="gpt-4o-mini",
        technique="harq_ir",
        max_rounds=3,
        estimated_cost_multiplier=3.0,
    ),
    ACMProfile(
        name="MCS-2: Careful",
        difficulty_range=(0.5, 0.7),
        model="gpt-4o",
        technique="turbo",
        max_rounds=4,
        estimated_cost_multiplier=6.0,
    ),
    ACMProfile(
        name="MCS-3: Redundant",
        difficulty_range=(0.7, 0.85),
        model="gpt-4o",
        technique="turbo",
        max_rounds=6,
        estimated_cost_multiplier=8.0,
    ),
    ACMProfile(
        name="MCS-4: Maximum Protection",
        difficulty_range=(0.85, 1.0),
        model="gpt-4o",
        technique="diversity_mrc",
        num_branches=3,
        estimated_cost_multiplier=12.0,
    ),
]


class ACMRouter:
    """
    Adaptive Coding & Modulation router.
    Estimates task difficulty, selects appropriate MCS profile, executes.
    """

    def __init__(
        self,
        channels: dict[str, AgentChannel],   # model_name → channel
        scorer: QualityScorer,
        acm_table: list[ACMProfile] | None = None,
        difficulty_estimator: AgentChannel | None = None,
        critic_channel: AgentChannel | None = None,
        category_tables: dict[str, list[ACMProfile]] | None = None,
    ):
        self.channels = channels
        self.scorer = scorer
        self.acm_table = acm_table or DEFAULT_ACM_TABLE
        # Optional per-category MCS tables. When task.category matches a key,
        # we route within that category's table; otherwise fall back to the
        # global acm_table. Category is a second CQI axis -- empirically,
        # task-type is a stronger predictor of oracle-winner than pilot
        # difficulty alone on mixed QA/reasoning/creative/code benchmarks.
        self.category_tables = category_tables or {}
        # Use a lightweight model for difficulty estimation
        self.difficulty_estimator = difficulty_estimator or next(iter(channels.values()))
        # Critic model used by HARQ/turbo profiles. Must match what plain
        # HARQ/turbo receive in the runner -- otherwise acm_harq_ir runs with
        # a weaker critic than harq_ir and underperforms on the same task.
        self.critic_channel = critic_channel

    def run(self, task: TaskItem) -> ReliabilityRun:
        # Step 1: Estimate task difficulty (channel estimation)
        difficulty, diff_output = self._estimate_difficulty(task)

        # Step 2: Select ACM profile. Prefer category-specific table when
        # available; fall back to the global difficulty-only table.
        cat = task.category.value if hasattr(task.category, "value") else str(task.category)
        table = self.category_tables.get(cat, self.acm_table)
        profile = self._select_profile(difficulty, table=table)

        logger.info(
            f"ACM routing: category={cat} difficulty={difficulty:.2f} → {profile.name} "
            f"({profile.technique}, model={profile.model})"
        )

        # Step 3: Execute selected technique
        run = self._execute_profile(task, profile, difficulty)
        # Track difficulty estimation as overhead cost
        run.overhead_outputs.append(diff_output)
        run.config["estimated_difficulty"] = difficulty
        run.config["selected_profile"] = profile.name
        run.config["routing_category"] = cat
        run.config["routing_mode"] = "category" if cat in self.category_tables else "global"
        # Record pilot-estimator provenance so plots/analysis can tell
        # pilot-logprob routing apart from self-rating fallback.
        if diff_output.mean_logprob is not None:
            run.config["difficulty_source"] = "pilot_logprob"
            run.config["difficulty_logprob"] = diff_output.mean_logprob
        else:
            run.config["difficulty_source"] = "self_rating"
        # Recompute metrics to include overhead
        run.compute_metrics()
        return run

    def _estimate_difficulty(self, task: TaskItem) -> tuple[float, AgentOutput]:
        """
        Estimate task difficulty on [0, 1] via a pilot probe.

        Communication analog: channel estimation from pilot symbols.
        We transmit a short probe with logprobs enabled; the mean token
        logprob of the model's own reply *is* the channel quality indicator:

            confidence = exp(mean_logprob)
            difficulty = 1 - confidence

        High model confidence on its own reply → good channel → low MCS.
        Low confidence → noisy channel → heavy protection.

        Falls back to an LLM self-rating (the old behavior) only when the
        backend does not expose logprobs (e.g. Anthropic SDK). This keeps
        cloud configs functional at the cost of breaking the pilot analogy.
        """
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

        # Fallback: LLM self-rating. Used only when logprobs are unavailable.
        prompt = (
            f"Rate the difficulty of this task on a scale from 0.0 to 1.0, "
            f"where 0.0 is trivial and 1.0 is extremely challenging.\n\n"
            f"Task category: {task.category.value}\n"
            f"Task: {task.prompt}\n\n"
            f'Respond with ONLY a JSON object: {{"difficulty": <float 0-1>, "reasoning": "<brief>"}}'
        )
        result = self.difficulty_estimator.transmit(prompt, temperature=0.1)

        try:
            import json
            text = QualityScorer._strip_thinking(result.text.strip())
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            return max(0.0, min(1.0, float(parsed.get("difficulty", 0.5)))), result
        except Exception:
            return 0.5, result

    def _select_profile(
        self, difficulty: float, table: list[ACMProfile] | None = None
    ) -> ACMProfile:
        """Select the MCS profile matching the estimated difficulty."""
        tbl = table if table is not None else self.acm_table
        for profile in tbl:
            low, high = profile.difficulty_range
            if low <= difficulty < high:
                return profile
        return tbl[-1]  # fallback to max protection

    def _execute_profile(self, task: TaskItem, profile: ACMProfile, difficulty: float) -> ReliabilityRun:
        """Execute the selected ACM profile's technique."""
        channel = self.channels.get(profile.model)
        if channel is None:
            # Fallback to first available channel
            channel = next(iter(self.channels.values()))

        if profile.technique == "uncoded":
            return self._run_uncoded(task, channel)
        elif profile.technique == "fec":
            svc = FECService(channel, self.scorer, code_rate=profile.code_rate)
            run = svc.run(task)
            run.technique = f"acm_fec_r{profile.code_rate}"
            return run
        elif profile.technique == "harq_ir":
            svc = HARQService(
                channel, self.scorer,
                mode=HARQMode.IR,
                max_rounds=profile.max_rounds,
                critic_channel=self.critic_channel,
            )
            run = svc.run(task)
            run.technique = "acm_harq_ir"
            return run
        elif profile.technique == "harq_cc":
            svc = HARQService(
                channel, self.scorer,
                mode=HARQMode.CC,
                max_rounds=profile.max_rounds,
                critic_channel=self.critic_channel,
            )
            run = svc.run(task)
            run.technique = "acm_harq_cc"
            return run
        elif profile.technique == "turbo":
            svc = TurboDecoder(
                generator=channel,
                critic=self.critic_channel or channel,
                scorer=self.scorer,
                max_iterations=max(profile.max_rounds, 2),
            )
            run = svc.run(task)
            run.technique = "acm_turbo"
            return run
        elif profile.technique == "fountain":
            channels_list = [channel] * max(profile.num_branches, 1)
            available = list(self.channels.values())
            if len(available) >= max(profile.num_branches, 2):
                channels_list = available[:max(profile.num_branches, 2)]
            svc = FountainDecoder(
                channels=channels_list,
                scorer=self.scorer,
            )
            run = svc.run(task)
            run.technique = "acm_fountain"
            return run
        elif profile.technique in ("diversity_mrc", "diversity_egc", "diversity_sc"):
            channels_list = [channel] * profile.num_branches
            available = list(self.channels.values())
            if len(available) >= profile.num_branches:
                channels_list = available[:profile.num_branches]
            combining_map = {
                "diversity_mrc": CombiningStrategy.MRC,
                "diversity_egc": CombiningStrategy.EGC,
                "diversity_sc":  CombiningStrategy.SC,
            }
            svc = DiversityEnsemble(
                channels_list, self.scorer,
                combining=combining_map[profile.technique],
            )
            run = svc.run(task)
            run.technique = f"acm_{profile.technique}"
            return run
        else:
            return self._run_uncoded(task, channel)

    def _run_uncoded(self, task: TaskItem, channel: AgentChannel) -> ReliabilityRun:
        """Simple uncoded transmission — single call, no redundancy."""
        run = ReliabilityRun(
            task_id=task.id,
            task_category=task.category.value,
            technique="acm_uncoded",
            config={"model": channel.model},
        )
        out = channel.transmit(task.request)
        out.quality_score = self.scorer.score(task.prompt, out.text, reference=task.reference, task=task)
        run.individual_outputs = [out]
        run.combined_output = out.text
        run.final_quality = out.quality_score
        run.compute_metrics()
        return run
