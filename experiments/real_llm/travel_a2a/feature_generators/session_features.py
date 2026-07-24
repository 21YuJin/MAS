"""
[Step 7B] Session Feature Group -- dispersion of per-call/per-message/
per-artifact metrics WITHIN one session, distinct from the session-level
TOTALS already defined in Phase 7A's raw_metadata_schema.json (event_count
etc.). latency_cv/output_token_cv/max_output_ratio inherit the same
mock-execution caveat as token_features/timing_features (Ollama-only raw
fields are None under mock); message_cv (message part_ids counts) and
artifact_cv (artifact sizes, from Part.size_bytes only -- never
Part.content) are structural and fully populated under mock.

No attack/difficulty/split/branch/content field is read anywhere in this
module.
"""
import statistics
from typing import Any, Dict, List, Optional

from ..models import AgentCallRecord, Artifact, Message, Part


def _cv(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    mean = statistics.mean(present)
    if mean == 0:
        return None
    return statistics.pstdev(present) / mean


def _max_ratio(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    mean = statistics.mean(present)
    if mean == 0:
        return None
    return max(present) / mean


def session_features_for_session(agent_call_records: List[AgentCallRecord], messages: List[Message],
                                  artifacts: List[Artifact], parts: List[Part]) -> Dict[str, Any]:
    latencies = [c.wall_clock_latency_ms for c in agent_call_records]
    output_tokens = [c.eval_count for c in agent_call_records]
    message_part_counts = [len(m.part_ids) for m in messages]
    size_by_part_id = {p.part_id: (p.size_bytes or 0) for p in parts}
    artifact_sizes = [sum(size_by_part_id.get(pid, 0) for pid in a.part_ids) for a in artifacts]

    return {
        "latency_cv": _cv(latencies),
        "output_token_cv": _cv(output_tokens),
        "message_cv": _cv(message_part_counts),
        "artifact_cv": _cv(artifact_sizes),
        "max_output_ratio": _max_ratio(output_tokens),
        "call_count": len(agent_call_records),
    }
