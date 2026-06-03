"""Privacy fuse: pin the set of keys that `build_event_from_result`
produces.

The day someone adds `prompt_hash`, `user_email`, or any other
plausibly-sensitive field thinking it's safe, this test fails. The
denylist scrubber in `agentcodec.telemetry._scrub` is the runtime
backstop; this is the unit-level backstop.

If you genuinely need to add a new field to the telemetry payload:

  1. Update README §"Anonymous telemetry" → the "What we send" table.
  2. Update NOTICE → the per-payload field list.
  3. Add the new key to PINNED_KEYS below.
  4. Get a second reviewer.
"""

from __future__ import annotations

import json

from agentcodec.telemetry import build_event_from_result

PINNED_KEYS = {
    "router_type",
    "technique_used",
    "task_category",
    "latency_s",
    "wall_clock_s",
    "cumulative_latency_s",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "rounds",
    "num_llm_calls",
    "thinking_used",
    # The (observed, predicted) pair is the retraining signal. `observed_*`
    # comes from the user's judge after dispatch; `predicted_*` comes from
    # the SemKNN q-matrix at /route time (only set on SemKNN routes).
    "observed_quality",
    "best_individual_quality",
    "diversity_gain",
    "observed_cost_usd",
    "judge_cost_usd",
    "cost_source",
    "lambda",
    "embedding",
    "embedding_bge_model",
    "user_config",
    "profile_used",
    "match_quality",
    "match_similarity",
    "estimate",
    "predicted_quality",
    "predicted_cost_usd",
    "k",
    "error_type",
}


class _FakeResult:
    """Mirror of the ReliabilityResult shape the builder reads."""
    technique_used = "harq_ir"
    latency_s = 2.34
    wall_clock_s = 2.45
    cumulative_latency_s = 2.45
    input_tokens = 1234
    output_tokens = 567
    thinking_tokens = 0
    rounds = 3
    num_llm_calls = 4
    thinking_used = False
    final_quality = 0.81
    best_individual_quality = 0.78
    diversity_gain = 0.03
    judge_cost_usd = 0.001
    cost_usd = 0.0123
    cost_source = "exact_user_rate"
    # Things that MUST NOT appear in the payload, even if read accidentally.
    text = "PROMPT OUTPUT — should never be sent"
    reference = "ground-truth answer"
    task_id = "private-task-id-with-pii"


def test_schema_pinned_against_drift() -> None:
    payload = build_event_from_result(
        result=_FakeResult(),
        routing_extra={
            "profile_used": "p", "match_quality": "exact",
            "match_similarity": 0.92,
            "predicted_quality_for_chosen": 0.78,
            "predicted_cost_for_chosen": 0.011,
            "estimate": False,
            "k": 20,
        },
        router_type="semknn_remote",
        user_config={"model_families": ["nemotron", "devstral"]},
        lambda_=5.0,
        embedding=[0.0] * 384,
        bge_model="BAAI/bge-small-en-v1.5",
        task_category="qa",
    )
    keys = set(payload.keys())
    new_keys = keys - PINNED_KEYS
    missing_keys = PINNED_KEYS - keys
    assert not new_keys, (
        "Telemetry payload grew new field(s) without an explicit privacy "
        f"review: {sorted(new_keys)}. See the test docstring for the "
        "playbook."
    )
    assert not missing_keys, (
        f"Telemetry payload lost expected field(s): {sorted(missing_keys)}. "
        "Adjust PINNED_KEYS if intentional."
    )


def test_payload_omits_forbidden_attributes() -> None:
    """Even when the result carries `text` / `reference` / `task_id` as
    attributes, the builder must not include them."""
    payload = build_event_from_result(
        result=_FakeResult(),
        routing_extra={},
        router_type="fixed",
        user_config=None,
        lambda_=None,
        embedding=None,
        bge_model=None,
        task_category="qa",
    )
    blob = json.dumps(payload)
    assert "PROMPT OUTPUT" not in blob
    assert "ground-truth" not in blob
    assert "private-task-id-with-pii" not in blob
    assert "text" not in payload
    assert "reference" not in payload
    assert "task_id" not in payload
