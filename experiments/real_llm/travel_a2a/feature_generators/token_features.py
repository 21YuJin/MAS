"""
[Step 7B] Token Feature Group -- derived from AgentCallRecord's Ollama
prompt_eval_count/eval_count fields (node-level raw metadata, Phase 7A).

Every value here is None whenever its underlying raw field is None -- most
notably, every field is None for every call under
timing_source == "deterministic_mock", since mock execution never calls
Ollama (mock_runner.py never sets llm_called=True or populates
prompt_eval_count/eval_count). This is expected, not a generator bug: real
values only exist once this workload is run through ollama_runtime
(configs/travel_a2a/formal_workload/formal_collection_plan.json, still
PLANNED_NOT_EXECUTED as of Step 7).

No attack/difficulty/split/branch/content field is read anywhere in this
module -- the only input type is AgentCallRecord, which structurally cannot
carry any of those (see models.py).
"""
from typing import Any, Dict, List, Optional

from ..models import AgentCallRecord


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return a / b


def _find_predecessor(call: AgentCallRecord,
                       calls_by_output_id: Dict[str, AgentCallRecord]) -> Optional[AgentCallRecord]:
    """The call whose output_part_ids/output_artifact_ids fed this call's
    input -- i.e. the upstream hop this call's prompt context grew from.
    Ambiguous fan-in (multiple producers) breaks ties by latest
    call_end_timestamp, then call_id, for determinism."""
    candidates = []
    for pid in call.input_part_ids:
        producer = calls_by_output_id.get(pid)
        if producer is not None and producer.call_id != call.call_id:
            candidates.append(producer)
    for aid in call.input_artifact_ids:
        producer = calls_by_output_id.get(aid)
        if producer is not None and producer.call_id != call.call_id:
            candidates.append(producer)
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (c.call_end_timestamp, c.call_id))[-1]


def token_features_for_session(agent_call_records: List[AgentCallRecord]) -> List[Dict[str, Any]]:
    calls_by_output_id: Dict[str, AgentCallRecord] = {}
    for call in agent_call_records:
        for pid in call.output_part_ids:
            calls_by_output_id[pid] = call
        for aid in call.output_artifact_ids:
            calls_by_output_id[aid] = call

    rows: List[Dict[str, Any]] = []
    for call in agent_call_records:
        input_token_count = call.prompt_eval_count
        output_token_count = call.eval_count
        total_token_count = (
            input_token_count + output_token_count
            if input_token_count is not None and output_token_count is not None else None
        )
        predecessor = _find_predecessor(call, calls_by_output_id)
        predecessor_output_token_count = predecessor.eval_count if predecessor is not None else None
        rows.append({
            "call_id": call.call_id,
            "agent_id": call.agent_id,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": total_token_count,
            "expansion_ratio": _safe_div(output_token_count, input_token_count),
            "token_difference": (
                output_token_count - input_token_count
                if input_token_count is not None and output_token_count is not None else None
            ),
            "predecessor_call_id": predecessor.call_id if predecessor is not None else None,
            "predecessor_output_ratio": _safe_div(predecessor_output_token_count, input_token_count),
            "ctx_delta": (
                input_token_count - predecessor_output_token_count
                if input_token_count is not None and predecessor_output_token_count is not None else None
            ),
        })
    return rows
