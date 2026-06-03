"""
15 — Code scoring: correctness + empirical time complexity.

Two ``score_mode`` values from :mod:`agentcodec.code_scoring`:

  * ``"code"``             — pass/fail correctness via test harness.
  * ``"code_complexity"``  — correctness + power-law fit of runtime vs.
                             input size, blended as
                             ``0.7 * correctness + 0.3 * complexity_credit``.

Both grade by actually executing the candidate. Default sandbox is a
fresh ``python -I`` subprocess with ``resource.setrlimit`` caps on
memory and CPU time; opt into Docker for hardened isolation by setting::

    AGENTCODEC_CODE_SANDBOX=docker

See the README §"Code scoring" for the full trade-off matrix.

This script calls the scorers directly (no LLM round-trip) so you can
see the scoring layer in isolation against fixed candidate strings.

Run::

    python examples/15_code_scoring.py
"""
from __future__ import annotations

from agentcodec.code_exec import _selected_backend, is_docker_available
from agentcodec.scoring import score_deterministic


# ----- HumanEval-style correctness fixture ----------------------------------

PALINDROME_METADATA = {
    "source": "humaneval",
    "entry_point": "is_palindrome",
    "test_code": (
        "def check(candidate):\n"
        "    assert candidate('') == True\n"
        "    assert candidate('aba') == True\n"
        "    assert candidate('aaaaa') == True\n"
        "    assert candidate('zbcd') == False\n"
        "    assert candidate('xywyx') == True\n"
        "    assert candidate('xywyz') == False\n"
        "    assert candidate('xywzx') == False\n"
    ),
}

CORRECT = """\
```python
def is_palindrome(text: str) -> bool:
    return text == text[::-1]
```
"""

BUGGY = """\
```python
def is_palindrome(text: str) -> bool:
    # Bug: only compares first half against its reverse
    n = len(text) // 2
    return text[:n] == text[:n][::-1]
```
"""

SYNTAX_ERR = """\
```python
def is_palindrome(text: str)
    return text == text[::-1]
```
"""


# ----- Complexity fixture: two valid algorithms, very different complexity --

# Candidate finds the maximum of a list. The fast version is O(n); the
# slow one is O(n^2). Both produce correct answers; complexity grading
# distinguishes them.
MAX_METADATA = {
    "source": "humaneval",                          # routes to "code"
    "entry_point": "find_max",
    "test_code": (
        "def check(candidate):\n"
        "    assert candidate([3, 1, 4, 1, 5, 9, 2, 6]) == 9\n"
        "    assert candidate([-1, -2, -3]) == -1\n"
        "    assert candidate([42]) == 42\n"
    ),
    # Size-parameterized runtime harness. ``setup`` runs once before
    # timing; ``call`` is interpolated with ``{N}`` for each measured
    # input size. The script emits ``COMPLEXITY:<N>:<seconds>`` per N.
    # Sizes chosen so the O(n^2) candidate fits in ~1 s at the largest N
    # (2700^2 ≈ 7.3 M ops × 3 repeats ≈ 0.25 s on a recent laptop).
    "complexity_inputs": {
        "sizes":   [100, 300, 900, 2700],
        "setup":   "import random; random.seed(0); "
                   "INPUTS = {n: [random.randint(0, 10**6) for _ in range(n)] "
                   "          for n in (100, 300, 900, 2700)}",
        "call":    "find_max(INPUTS[{N}])",
        "repeats": 5,
        "timeout_s": 60,
    },
}

FAST_MAX = """\
```python
def find_max(xs):
    m = xs[0]
    for x in xs:
        if x > m:
            m = x
    return m
```
"""

# Genuinely O(n^2): nested loops with no short-circuiting.
SLOW_MAX = """\
```python
def find_max(xs):
    m = xs[0]
    for i in range(len(xs)):
        for j in range(len(xs)):
            if xs[j] > m:
                m = xs[j]
    return m
```
"""


def _score(mode: str, label: str, output: str, metadata: dict) -> None:
    score = score_deterministic(mode, output=output, reference="", metadata=metadata)
    print(f"  {label:<24} → {score:.3f}")


def main() -> None:
    print(f"Sandbox backend : {_selected_backend()}")
    print(f"Docker on PATH  : {is_docker_available()}")
    print()
    print(
        "Set AGENTCODEC_CODE_SANDBOX=docker to flip the backend.\n"
        "Other knobs: AGENTCODEC_CODE_TIMEOUT_S, AGENTCODEC_CODE_MEMORY_MB."
    )
    print()

    print('--- score_mode="code" (pass/fail correctness) ---')
    print('Task: implement is_palindrome.')
    _score("code", "correct answer",      CORRECT,    PALINDROME_METADATA)
    _score("code", "buggy but parseable", BUGGY,      PALINDROME_METADATA)
    _score("code", "syntax error",        SYNTAX_ERR, PALINDROME_METADATA)

    print()
    print('--- score_mode="code_complexity" (correctness + Big-O fit) ---')
    print('Task: implement find_max — two correct algorithms, very different complexity.')
    _score("code_complexity", "O(n) linear scan",     FAST_MAX, MAX_METADATA)
    _score("code_complexity", "O(n²) nested loops",   SLOW_MAX, MAX_METADATA)

    print()
    print("Interpretation:")
    print("  code:            1.00 = all assertions passed inside the budget")
    print("                   0.50 = parses but unverifiable / sandbox failed")
    print("                   0.00 = syntax error, failed assertion, or timeout")
    print()
    print("  code_complexity: blended as 0.7 * correctness + 0.3 * complexity_credit.")
    print("                   O(1) / O(log n) / O(n) get ≥0.9 complexity_credit;")
    print("                   O(n²) gets 0.5; O(n³) gets 0.3; worse gets 0.1.")
    print("                   Wrong answers always score 0, regardless of speed.")


if __name__ == "__main__":
    main()
