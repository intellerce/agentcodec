"""
02 — Self-refine, scored by actually running the code.

The model writes a draft, critiques it, revises, then loops up to
``max_rounds`` times. One model, no judge ensemble — the same channel plays
generator AND critic.

This example targets a code task and attaches a test fixture, so the
score is execution-based, not just the judge's opinion: each draft is run
in the sandbox against the ``check()`` asserts (``metadata.test_code``),
and ``final_quality`` blends that pass/fail signal with the judge
(``0.6 × {0, 0.5, 1} + 0.4 × judge``). Set the backend via
``$AGENTCODEC_CODE_SANDBOX`` (default ``subprocess``; ``docker`` opt-in).
For a deeper dive on code scoring and a harq_ir comparison, see
``16_code_scoring_end_to_end.py``.

When it shines:
  - Free-form prose or code where the first draft is decent but rushed.
  - Tasks the model genuinely *can* solve but routinely under-thinks.

When it doesn't help:
  - Pure-recall QA (the second draft just rephrases the first).
  - Tasks where the model is wrong from the start — no critic in the room.

Run:
    python examples/02_self_refine.py
"""
from __future__ import annotations

from agentcodec import ReliabilityModule
from agentcodec.code_exec import _selected_backend

from _common import explain_score, judge_block, model_block, print_result


# Pin the signature in a fenced block so the scorer's code extractor gets a
# clean function definition (same convention as example 16).
PROMPT = """\
Implement the following Python function. Return only the function inside a
fenced ```python code block — no commentary, no usage examples.

```python
def is_balanced(s: str) -> bool:
    \"\"\"Return True iff every '(' '{' '[' in `s` has a matching close in
    the correct order. Characters other than brackets are ignored.
    >>> is_balanced('()[]{}')
    True
    >>> is_balanced('([)]')
    False
    \"\"\"
```
"""

# source="humaneval" makes the library auto-infer score_mode="code", which
# routes scoring into the sandbox. `entry_point` is the symbol `check()` is
# called with; `test_code` is the pass/fail harness run against each draft.
METADATA = {
    "source":      "humaneval",
    "entry_point": "is_balanced",
    "test_code": (
        "def check(candidate):\n"
        "    assert candidate('') == True\n"
        "    assert candidate('()[]{}') == True\n"
        "    assert candidate('{[]}') == True\n"
        "    assert candidate('([{}])') == True\n"
        "    assert candidate('a(b)c[d]') == True\n"
        "    assert candidate('(]') == False\n"
        "    assert candidate('([)]') == False\n"
        "    assert candidate('(') == False\n"
        "    assert candidate(']') == False\n"
        "    assert candidate('((())') == False\n"
    ),
}


def main() -> None:
    print(f"Sandbox backend: {_selected_backend()}")
    mod = ReliabilityModule.from_dict({
        "models": [model_block("qwen3:8b", temperature=0.6)],
        "judge": judge_block(),
        "strategy": {
            "type": "fixed",
            "technique": "self_refine",
            # max_rounds is the only knob. 1 = just the draft; 3 = draft + 2 revisions.
            "params": {"max_rounds": 3},
        },
        "defaults": {"category": "auto"},
    })

    with mod:
        result = mod.run(
            prompt=PROMPT,
            reference="",        # unused for code; the test fixture is the source of truth
            category="code",
            metadata=METADATA,   # source="humaneval" → score_mode="code" → sandbox
            return_trace=True,   # needed for the per-criterion breakdown below
        )
        print_result(result, label="self_refine x3", score_mode="code")
        # Detailed breakdown: sandbox pass/fail blended with the judge's
        # 15-criterion checklist on the final refined answer.
        explain_score(result, score_mode="code")


if __name__ == "__main__":
    main()
