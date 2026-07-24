"""
[Step 6-9/6-10/6-11] Metadata delta summary -- compares raw AgentCallRecord/
InteractionEvent/Artifact aggregates between a matched normal/attack pair.

Deliberately NOT feature engineering: no correlation filtering, no z-score,
no scaler fitting, no candidate-feature selection happens here -- just
side-by-side normal_value/attack_value/absolute_delta/relative_delta on
already-existing raw fields (the same raw fields Step 4's AgentCallRecord and
Step 2's InteractionEvent/Artifact already store), grouped by which
outcome_group (Step 6-11) the pair landed in.

The point of outcome grouping is exploratory: does a metadata difference
appear only once the attack has SOME observable effect, or does even a
"no_effect" injection-present session already look different from its
matched normal? That question matters for later interpretation of what
LightGAE is actually keying on -- it is not answered here, only set up for.
"""
import statistics
from typing import Any, Dict, List, Optional

OUTCOME_GROUPS = ("no_effect", "entry_effect_only", "propagated_effect", "successful_goal")


def classify_outcome_group(diagnostics: dict) -> str:
    """[Step 6-11] Derived purely from fields the evaluator already computed
    -- no new information, just a coarser bucket for aggregation across many
    sessions. Order matters: goal_success implies propagation/entry effects
    already happened, so it's checked first."""
    if diagnostics.get("goal_success"):
        return "successful_goal"
    if diagnostics.get("propagation_observed"):
        return "propagated_effect"
    if (diagnostics.get("entry_agent_exposed") or diagnostics.get("instruction_followed")
            or diagnostics.get("indicator_observed")):
        return "entry_effect_only"
    return "no_effect"


def _agent_call_aggregate(agent_call_records: List) -> Dict[str, Any]:
    llm_records = [r for r in agent_call_records if r.llm_called]

    def _mean(key):
        vals = [getattr(r, key) for r in llm_records if getattr(r, key) is not None]
        return statistics.mean(vals) if vals else None

    call_count_by_agent: Dict[str, int] = {}
    for r in agent_call_records:
        call_count_by_agent[r.agent_id] = call_count_by_agent.get(r.agent_id, 0) + 1

    return {
        "prompt_eval_count_mean": _mean("prompt_eval_count"), "eval_count_mean": _mean("eval_count"),
        "prompt_eval_duration_mean": _mean("prompt_eval_duration"), "eval_duration_mean": _mean("eval_duration"),
        "total_duration_mean": _mean("total_duration"), "load_duration_mean": _mean("load_duration"),
        "wall_clock_latency_ms_mean": _mean("wall_clock_latency_ms"),
        "call_count_by_agent": call_count_by_agent,
        "total_call_count": len(agent_call_records), "llm_call_count": len(llm_records),
    }


def _event_aggregate(events: List, messages: List) -> Dict[str, Any]:
    unique_pairs = {(e.sender_id, e.receiver_id) for e in events}
    revision_count = sum(1 for m in messages if m.interaction_type.value == "revision_request")
    clarification_count = sum(1 for m in messages if m.interaction_type.value == "clarification_request")
    status_transition_count = sum(1 for e in events if e.status_before != e.status_after)
    return {
        "event_count": len(events), "unique_directed_pair_count": len(unique_pairs),
        "message_count": len(messages), "revision_request_count": revision_count,
        "clarification_request_count": clarification_count, "status_transition_count": status_transition_count,
    }


def _artifact_aggregate(artifacts: List, parts: List) -> Dict[str, Any]:
    return {
        "artifact_count": len(artifacts),
        "artifact_version_count": sum(a.version for a in artifacts),
        "artifact_size_bytes_total": sum((p.size_bytes or 0) for p in parts),
        "record_count_total": sum(a.record_count for a in artifacts),
    }


def _delta(normal_value, attack_value) -> Dict[str, Optional[float]]:
    if normal_value is None or attack_value is None or isinstance(normal_value, dict):
        return {"normal_value": normal_value, "attack_value": attack_value,
                "absolute_delta": None, "relative_delta": None}
    absolute_delta = attack_value - normal_value
    relative_delta = (absolute_delta / normal_value) if normal_value else None
    return {"normal_value": normal_value, "attack_value": attack_value,
            "absolute_delta": absolute_delta, "relative_delta": relative_delta}


_AGENT_CALL_KEYS = ("prompt_eval_count_mean", "eval_count_mean", "prompt_eval_duration_mean", "eval_duration_mean",
                     "total_duration_mean", "load_duration_mean", "wall_clock_latency_ms_mean",
                     "total_call_count", "llm_call_count")
_EVENT_KEYS = ("event_count", "unique_directed_pair_count", "message_count", "revision_request_count",
               "clarification_request_count", "status_transition_count")
_ARTIFACT_KEYS = ("artifact_count", "artifact_version_count", "artifact_size_bytes_total", "record_count_total")


def compute_metadata_delta_summary(normal_result, attack_result, diagnostics: dict) -> Dict[str, Any]:
    normal_calls = _agent_call_aggregate(normal_result.agent_call_records)
    attack_calls = _agent_call_aggregate(attack_result.agent_call_records)
    normal_events = _event_aggregate(normal_result.events, normal_result.messages)
    attack_events = _event_aggregate(attack_result.events, attack_result.messages)
    normal_artifacts = _artifact_aggregate(normal_result.artifacts, normal_result.parts)
    attack_artifacts = _artifact_aggregate(attack_result.artifacts, attack_result.parts)

    deltas = {}
    for key in _AGENT_CALL_KEYS:
        deltas[f"agent_call.{key}"] = _delta(normal_calls[key], attack_calls[key])
    for key in _EVENT_KEYS:
        deltas[f"event.{key}"] = _delta(normal_events[key], attack_events[key])
    for key in _ARTIFACT_KEYS:
        deltas[f"artifact.{key}"] = _delta(normal_artifacts[key], attack_artifacts[key])

    return {
        "outcome_group": classify_outcome_group(diagnostics),
        "deltas": deltas,
        "call_count_by_agent": {"normal": normal_calls["call_count_by_agent"],
                                 "attack": attack_calls["call_count_by_agent"]},
    }
