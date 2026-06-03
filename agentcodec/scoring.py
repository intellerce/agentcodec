"""
Deterministic scoring helpers for benchmarks with structured references.

The LLM judge ([`QualityScorer`](channel.py)) is designed for free-form answers
where partial credit and rubric grading make sense. For benchmarks whose
reference is a single letter, a yes/no, a number, or a short normalized
phrase, the judge collapses to binary correctness anyway — and adds noise
plus latency plus cost. The helpers here let those tasks bypass the judge
entirely with mode-specific deterministic checks.

Each public scorer returns a float in [0, 1]. Most return 0.0 or 1.0; the
"relaxed" mode can return values for partial numeric matches (within
tolerance) if you want to extend it later.

Score modes (set on `TaskItem.score_mode`):
    "exact_letter" — extract A-J from output, compare to reference letter
    "exact_match"  — case-insensitive normalized equality
    "yes_no"       — boolean extraction + match
    "numeric"      — parse number, exact match modulo formatting
    "relaxed"      — numeric within 5%, fallback to string equality (ChartQA-style)
    "judge"        — defer to the LLM judge (same as None)

Statistical helpers:
    wilson_ci(successes, total, confidence) → (low, high)
    accuracy(scores) → (mean, lo, hi, n)
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Output extractors
# ---------------------------------------------------------------------------

# Letters used as MC option labels. 10 is enough for every benchmark we ship.
_VALID_LETTERS = "ABCDEFGHIJ"


def extract_letter(output: str, valid: str = _VALID_LETTERS) -> str | None:
    """
    Pull a single MC-option letter out of `output`. Tolerates the common
    answer phrasings models emit:
        "C", "(C)", "C.", "C)", "Answer: C", "The answer is C", "C: ...",
        markdown like "**C**".
    Returns the uppercased letter or None if no plausible match was found.
    """
    if not output:
        return None
    text = output.strip()

    # Special case: the entire stripped output is a single MC letter. Accept
    # it regardless of case (handles models that emit a bare "c" / "C" with
    # no surrounding text).
    if len(text) == 1 and text.upper() in valid:
        return text.upper()

    # Strongest signals first: explicit "answer is X" / "answer: X" phrases.
    # Each entry is (regex, flags, pick_last). pick_last=True for the
    # last-resort scanner so we get the final-answer letter at the end of
    # a chain of reasoning rather than the first incidental match.
    patterns = [
        # "Answer: X" / "answer is X" / "final answer is X". Case-insensitive
        # so we catch "answer: c" — letter extraction handles uppercasing.
        (rf"\b(?:final\s+)?answer\s*(?:is|:)\s*\(?\s*([{valid}])\s*\)?\b",
         re.IGNORECASE, False),
        # Starts the response with a clearly-terminated letter: "X." / "X)"
        # / "X:" / "(X)" / "**X**.". We don't accept "X " (space) because
        # sentence-leading capital words ("I think...", "A few options...")
        # would steal priority. Those fall through to pattern 3.
        (rf"^\s*\*{{0,2}}\(?\s*([{valid}])\s*\)?\*{{0,2}}\s*[\.\):]",
         re.IGNORECASE, False),
        # Last resort: any standalone letter token. CASE-SENSITIVE (no
        # IGNORECASE) to avoid matching the English articles "a" and "i" as
        # MC letters — every MC model in practice emits uppercase answers.
        (rf"(?<![A-Za-z])([{valid}])(?![A-Za-z])", 0, True),
    ]
    for pat, flags, pick_last in patterns:
        matches = list(re.finditer(pat, text, flags))
        if not matches:
            continue
        m = matches[-1] if pick_last else matches[0]
        return m.group(1).upper()
    return None


def extract_yes_no(output: str) -> bool | None:
    """
    Resolve `output` to True (yes) / False (no) / None (unparseable).
    Looks for "yes" / "no" / "true" / "false" tokens, case-insensitive.
    If both appear, prefers the LAST one (matches "the answer is no" → no).
    """
    if not output:
        return None
    text = output.lower()
    yes_re = re.compile(r"\b(yes|true|correct|right)\b")
    no_re = re.compile(r"\b(no|false|incorrect|wrong)\b")
    yes_iter = list(yes_re.finditer(text))
    no_iter = list(no_re.finditer(text))
    if not yes_iter and not no_iter:
        return None
    last_yes = yes_iter[-1].start() if yes_iter else -1
    last_no = no_iter[-1].start() if no_iter else -1
    return last_yes > last_no


_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def extract_number(text: str) -> float | None:
    """
    Pull a single number from `text`. Strips currency / percent signs and
    thousands commas. Returns the LAST number found, since models tend to
    state intermediate values before the final answer.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = re.sub(r"[\$€£¥]", "", str(text))
    matches = _NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation."""
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    s = re.sub(r"[\.\,\!\?\:\;]+$", "", s)
    return s


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------

def score_exact_letter(output: str, reference: str) -> float:
    """Multi-choice: 1.0 if extracted letter matches the reference letter."""
    ref_letter = extract_letter(reference)
    if ref_letter is None:
        # Reference itself is unparseable — fall back to substring equality
        ref_letter = (reference or "").strip().upper()[:1]
    got = extract_letter(output)
    return 1.0 if got is not None and got == ref_letter.upper() else 0.0


def score_yes_no(output: str, reference: str) -> float:
    """Yes/no: 1.0 on agreement of boolean extraction."""
    ref_bool = extract_yes_no(reference)
    if ref_bool is None:
        # Reference is a literal "yes" / "no" string; force it
        rs = _normalize_text(reference)
        if rs.startswith("y") or rs in ("true", "1"):
            ref_bool = True
        elif rs.startswith("n") or rs in ("false", "0"):
            ref_bool = False
        else:
            return 0.0
    out_bool = extract_yes_no(output)
    return 1.0 if out_bool is not None and out_bool == ref_bool else 0.0


def score_exact_match(output: str, reference: str) -> float:
    """
    Case-insensitive normalized equality. Also accepts the case where the
    reference appears as a substring of the output's tail (typical for
    "The answer is <reference>." patterns).
    """
    out_n = _normalize_text(output)
    ref_n = _normalize_text(reference)
    if not ref_n:
        return 0.0
    if out_n == ref_n:
        return 1.0
    # Tail-substring match: reference appears near the end of output.
    tail = out_n[-max(len(ref_n) * 4, 200):]
    if ref_n in tail:
        return 1.0
    return 0.0


def score_numeric(output: str, reference: str, rel_tol: float = 0.0) -> float:
    """
    Parse a number from each side and compare. rel_tol=0 → exact match
    modulo formatting (commas, currency, trailing zeros). rel_tol > 0 →
    accept if |out − ref| ≤ rel_tol * max(|ref|, 1).
    """
    ref = extract_number(reference)
    got = extract_number(output)
    if ref is None or got is None:
        return 0.0
    if rel_tol <= 0:
        # Exact match modulo float precision (1e-9 absolute slack).
        return 1.0 if math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-9) else 0.0
    threshold = rel_tol * max(abs(ref), 1.0)
    return 1.0 if abs(got - ref) <= threshold else 0.0


def score_relaxed(output: str, reference: str, rel_tol: float = 0.05) -> float:
    """
    ChartQA-style "relaxed accuracy":
    - if the reference parses as a number, do tolerance match (default 5%);
    - otherwise, case-insensitive normalized string equality.
    """
    ref_num = extract_number(reference)
    if ref_num is not None:
        return score_numeric(output, reference, rel_tol=rel_tol)
    return score_exact_match(output, reference)


def _score_code_lazy(output: str, reference: str, metadata: dict | None = None) -> float:
    """Lazy proxy — defer importing ``code_scoring`` until first call.

    Avoids paying for the sandbox lookup and the (small) ``ast`` import
    on every process startup that doesn't grade code.
    """
    from .code_scoring import score_code
    return score_code(output, reference, metadata)


def _score_code_complexity_lazy(
    output: str, reference: str, metadata: dict | None = None,
) -> float:
    """Lazy proxy for ``score_code_complexity``."""
    from .code_scoring import score_code_complexity
    return score_code_complexity(output, reference, metadata)


_DISPATCH = {
    "exact_letter": score_exact_letter,
    "exact_match": score_exact_match,
    "yes_no": score_yes_no,
    "numeric": score_numeric,
    "relaxed": score_relaxed,
    "code": _score_code_lazy,
    "code_complexity": _score_code_complexity_lazy,
}

# Scorers that need access to the full task metadata (test fixtures,
# complexity inputs, entry-point name, …) rather than just
# (output, reference). They are called with a 3-arg signature; everything
# else stays on 2-arg.
_METADATA_AWARE = frozenset({"code", "code_complexity"})


def score_deterministic(
    mode: str,
    output: str,
    reference: str,
    metadata: dict | None = None,
) -> float:
    """
    Single entry point for deterministic scoring. Raises KeyError on an
    unknown mode so QualityScorer.score() surfaces typos immediately
    instead of silently defaulting to the judge.

    ``metadata`` is consulted only for modes in :data:`_METADATA_AWARE`
    (currently just ``"code"``, which needs the test fixture). Other
    scorers ignore it — kept on the signature so call sites don't have
    to special-case which mode they're invoking.
    """
    fn = _DISPATCH[mode]
    if mode in _METADATA_AWARE:
        return fn(output, reference, metadata)
    return fn(output, reference)


SUPPORTED_MODES = frozenset(_DISPATCH.keys()) | {"judge"}


# ---------------------------------------------------------------------------
# Inference from task metadata
# ---------------------------------------------------------------------------

# Curated map: metadata.source → score_mode for benchmarks we know the shape
# of. Keep this narrow: false inferences are worse than missing inferences,
# because they silently change scoring regimes.
_SOURCE_SCORE_MODES: dict[str, str] = {
    # Multi-choice
    "mmlu": "exact_letter",
    "arc": "exact_letter",
    "arc_easy": "exact_letter",
    "arc_challenge": "exact_letter",
    "hellaswag": "exact_letter",
    "winogrande": "exact_letter",
    "openbookqa": "exact_letter",
    "commonsenseqa": "exact_letter",
    "truthfulqa_mc": "exact_letter",
    "mmlu_pro": "exact_letter",
    # Yes/no
    "boolq": "yes_no",
    # Numeric
    "gsm8k": "numeric",
    "math": "numeric",
    "mathqa": "numeric",
    "asdiv": "numeric",
    "svamp": "numeric",
    # ChartQA-style: number-with-tolerance OR short string
    "chartqa": "relaxed",
    # Code — execution-based scoring via Docker sandbox. Requires the
    # task to carry ``metadata.test_code`` (the HumanEval check harness)
    # and (typically) ``metadata.entry_point``. Falls back to a
    # syntax-only grade when Docker isn't on $PATH.
    "humaneval": "code",
    "mbpp": "code",
}


def infer_score_mode_from_metadata(metadata: dict | None) -> str | None:
    """
    Infer a `score_mode` for tasks that didn't set one explicitly.

    Keyed off `metadata["source"]` (case-insensitive) against a curated
    table of known benchmark datasets. Returns None for unknown or
    free-form sources — those keep the pure-judge default.

    The lookup is intentionally narrow: a false inference is worse than
    no inference, because it silently changes the scoring regime. Add a
    new source here only when its reference shape is unambiguous.
    """
    if not metadata:
        return None
    source = str(metadata.get("source") or "").strip().lower()
    if not source:
        return None
    return _SOURCE_SCORE_MODES.get(source)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a Bernoulli proportion. Returns
    (low, high) bounded to [0, 1]. Preferred over the normal-approx CI
    because it has correct coverage even at extreme proportions and small
    samples — the regime every per-task-pool accuracy estimate lives in.
    """
    if total <= 0:
        return (0.0, 0.0)
    # Two-sided z for the given confidence level. Inverse of the standard
    # normal CDF via a Beasley-Springer-Moro-style rational approximation
    # would be overkill — use a tiny lookup keyed on the common levels.
    z_table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.5758}
    z = z_table.get(confidence, 1.9600)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2 * total)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy(scores: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, int]:
    """
    Aggregate per-task scores into (mean, ci_low, ci_high, n).

    Treats scores ≥ 0.5 as a "success" for the binomial CI — this is the
    canonical reading when scores live in {0, 1} (which is what the
    deterministic modes produce). For mixed binary/continuous pools the
    mean is still reported correctly; only the CI is approximate.
    """
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    mean = sum(scores) / n
    successes = sum(1 for s in scores if s >= 0.5)
    lo, hi = wilson_ci(successes, n, confidence=confidence)
    return (mean, lo, hi, n)
