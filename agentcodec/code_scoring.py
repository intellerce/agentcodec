"""
Code scoring — HumanEval-style execution-based correctness + empirical
time-complexity grading.

Two modes plug into :data:`agentcodec.scoring._DISPATCH`:

  * ``score_mode="code"``             — pass/fail correctness only.
  * ``score_mode="code_complexity"``  — correctness + empirical Big-O
                                        fit, blended into one score.

Both grade by actually running the candidate in
:func:`agentcodec.code_exec.run_sandboxed`, which selects the subprocess
or Docker backend based on ``$AGENTCODEC_CODE_SANDBOX``. The default is
the subprocess backend (no extra dependencies); set the env var to
``docker`` for hardened isolation. See the project README for the full
trade-off matrix.

Wiring: ``"humaneval"`` and ``"mbpp"`` auto-route to ``"code"`` via
:data:`agentcodec.scoring._SOURCE_SCORE_MODES`. ``"code_complexity"``
is opt-in — there's no curated benchmark that ships size-parameterized
fixtures, so it must be selected explicitly via ``score_mode=`` on the
task.
"""

from __future__ import annotations

import ast
import logging
import math
import re
import statistics
from typing import Any

from .code_exec import ExecutionResult, run_sandboxed

logger = logging.getLogger(__name__)


# --- Code extraction ------------------------------------------------------

def _extract_code(output: str) -> str:
    """Pull a Python code block out of a free-form LLM reply.

    Order of preference:

      1. A fenced ``​python … ​`` block.
      2. A fenced ``​ … ​`` block with no language tag.
      3. The whole reply, stripped.

    We deliberately avoid splicing multiple fenced blocks: tasks in the
    corpus expect a single function definition, and merging blocks can
    produce broken code.
    """
    fenced = re.search(r"```python\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced.group(1).rstrip()
    fenced = re.search(r"```\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced.group(1).rstrip()
    return output.strip()


def _syntax_ok(code: str) -> bool:
    """Return True iff ``code`` parses as a Python module."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# --- Correctness (Phase 1) ------------------------------------------------

def _build_test_harness(code: str, metadata: dict[str, Any]) -> str:
    """Assemble: candidate code + test_code + check(entry_point)."""
    parts = [code, metadata["test_code"]]
    entry_point = metadata.get("entry_point")
    if entry_point:
        parts.append(f"check({entry_point})")
    return "\n\n".join(parts)


def _log_sandbox_failure(label: str, result: ExecutionResult) -> None:
    if result.error:
        logger.warning("score_code: %s (%s backend): %s",
                       label, result.backend, result.error)


def score_code(
    output: str,
    reference: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Grade a code answer by executing it against the task's test fixture.

    Returns a float in [0, 1]:

      * ``1.0`` — code parsed, harness ran, exit code 0 within budget.
      * ``0.0`` — syntax error, harness failure, non-zero exit, or timeout.
      * ``0.5`` — parses but unverifiable (no ``test_code`` on the task,
        or the sandbox harness itself failed). Treated as partial credit
        so a clearly broken answer still scores worse.

    ``reference`` is accepted for interface symmetry with the other
    deterministic scorers but is unused: the test fixture in
    ``metadata`` is the source of truth for correctness.
    """
    code = _extract_code(output)
    if not _syntax_ok(code):
        return 0.0

    if not metadata or not metadata.get("test_code"):
        return 0.5  # parseable but unverifiable

    program = _build_test_harness(code, metadata)
    result = run_sandboxed(program)
    if result.error:
        _log_sandbox_failure("sandbox failed", result)
        return 0.5  # don't claim correctness; degrade
    if result.timed_out:
        return 0.0
    return 1.0 if result.exit_code == 0 else 0.0


# --- Complexity (Phase 2) -------------------------------------------------

# Power-law thresholds for log-log fit ``log(t) = k * log(N) + c``.
# (lower_bound_inclusive, label, quality_credit)
_COMPLEXITY_TIERS: tuple[tuple[float, str, float], ...] = (
    (-math.inf, "O(1)",        1.00),
    (0.20,      "O(log n)",    1.00),
    (0.70,      "O(n)",        0.90),
    (1.30,      "O(n log n)",  0.85),
    (1.70,      "O(n^2)",      0.50),
    (2.30,      "O(n^3)",      0.30),
    (3.30,      "O(n^k>=4) or exponential", 0.10),
)


def _classify(exponent: float) -> tuple[str, float]:
    """Pick a tier for the fitted exponent."""
    label, credit = _COMPLEXITY_TIERS[0][1], _COMPLEXITY_TIERS[0][2]
    for lo, lbl, cr in _COMPLEXITY_TIERS:
        if exponent >= lo:
            label, credit = lbl, cr
        else:
            break
    return label, credit


def _fit_power_law(sizes: list[int], timings: list[float]) -> float | None:
    """Least-squares fit ``log(t) = k * log(N) + c``; return ``k`` or None.

    Returns ``None`` if the points are degenerate (all same N, no
    variance) — the caller treats that as "unable to classify".
    """
    if len(sizes) < 2 or len(sizes) != len(timings):
        return None
    log_n = [math.log(max(n, 1)) for n in sizes]
    log_t = [math.log(max(t, 1e-9)) for t in timings]
    mean_n = statistics.fmean(log_n)
    mean_t = statistics.fmean(log_t)
    num = sum((log_n[i] - mean_n) * (log_t[i] - mean_t) for i in range(len(log_n)))
    den = sum((log_n[i] - mean_n) ** 2 for i in range(len(log_n)))
    if den == 0:
        return None
    return num / den


def _build_complexity_harness(
    code: str,
    metadata: dict[str, Any],
    sizes: list[int],
    repeats: int,
) -> str:
    """Assemble a one-shot program that times the candidate on each N.

    Emits ``COMPLEXITY:<N>:<median_seconds>`` for each size, parsed by
    the host. The ``setup`` and ``call`` from
    ``metadata["complexity_inputs"]`` are interpolated with ``{N}``.
    """
    inputs = metadata["complexity_inputs"]
    setup = inputs.get("setup", "")
    call = inputs["call"]  # required: a Python expression that calls the candidate
    harness = [
        code,
        "import time, statistics",
        setup,
        f"_SIZES = {sizes!r}",
        f"_REPEATS = {repeats}",
        "for _N in _SIZES:",
        "    _ts = []",
        "    for _ in range(_REPEATS):",
        "        _t0 = time.perf_counter()",
        f"        {call.format(N='_N')}",
        "        _ts.append(time.perf_counter() - _t0)",
        "    print(f'COMPLEXITY:{_N}:{statistics.median(_ts)}')",
    ]
    return "\n".join(harness)


def _parse_timings(stdout: str) -> dict[int, float]:
    """Extract ``COMPLEXITY:<N>:<t>`` lines from harness stdout."""
    out: dict[int, float] = {}
    for line in stdout.splitlines():
        m = re.match(r"COMPLEXITY:(\d+):([\deE.+\-]+)$", line.strip())
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def score_code_complexity(
    output: str,
    reference: str,
    metadata: dict[str, Any] | None = None,
) -> float:
    """Score code on correctness + empirical time complexity.

    Pipeline:

      1. Run :func:`score_code` for correctness (0 / 0.5 / 1).
      2. If correctness is 0, return 0 — wrong code gets no complexity
         credit no matter how fast it is.
      3. If the task has no ``complexity_inputs``, return ``correctness``
         unchanged (no complexity dimension to grade on).
      4. Otherwise, run the candidate on each ``sizes`` value, take the
         median of ``repeats`` runs per size, fit a power law to
         ``(log N, log t)``, classify into a tier, and return
         ``0.7 * correctness + 0.3 * complexity_credit``.

    Required metadata shape::

        metadata["complexity_inputs"] = {
            "sizes":   [100, 1000, 10000],
            "setup":   "<optional> Python statements run once before timing",
            "call":    "<entry_point>(make_input({N}))",
            "repeats": 3,                      # optional, default 3
        }

    The ``call`` string is ``.format(N=<size>)``-substituted, so use
    literal ``{N}`` placeholders for the size variable.
    """
    correctness = score_code(output, reference, metadata)
    if correctness == 0.0:
        return 0.0
    if not metadata or "complexity_inputs" not in metadata:
        return correctness

    inputs = metadata["complexity_inputs"]
    sizes = list(inputs.get("sizes") or [])
    if len(sizes) < 2 or "call" not in inputs:
        logger.warning(
            "score_code_complexity: malformed complexity_inputs "
            "(need >=2 sizes and a 'call' string); skipping complexity."
        )
        return correctness
    repeats = int(inputs.get("repeats", 3))

    code = _extract_code(output)
    program = _build_complexity_harness(code, metadata, sizes, repeats)

    # Complexity runs typically need a larger time budget than a single
    # correctness check (we run the function len(sizes) * repeats times).
    budget = float(inputs.get("timeout_s", 30))
    result = run_sandboxed(program, timeout_s=budget)
    if result.error:
        _log_sandbox_failure("complexity sandbox failed", result)
        return correctness  # don't penalise — sandbox is the issue
    if not result.ok:
        # Either timeout or non-zero exit. Conservatively treat as "no
        # complexity signal" rather than rebranding the code as wrong.
        return 0.7 * correctness  # mild penalty for unmeasurable timing

    timings = _parse_timings(result.stdout)
    if len(timings) != len(sizes):
        logger.warning(
            "score_code_complexity: expected %d timing lines, got %d. "
            "Check that the harness `call` actually executes for every size.",
            len(sizes), len(timings),
        )
        return 0.7 * correctness

    ordered_sizes = sorted(timings.keys())
    ordered_times = [timings[n] for n in ordered_sizes]
    k = _fit_power_law(ordered_sizes, ordered_times)
    if k is None:
        return 0.7 * correctness
    _, credit = _classify(k)
    return 0.7 * correctness + 0.3 * credit


__all__ = ["score_code", "score_code_complexity"]
