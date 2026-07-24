"""
[Step 7B] Timing Feature Group -- derived from AgentCallRecord's Ollama
duration fields (node-level, nanoseconds) plus wall_clock_latency_ms.

wall_clock_latency_ms is populated under BOTH timing_source values (mock uses
a synthetic-but-real millisecond diff from DeterministicClock). Every other
field here (generation_time_ms, prompt_eval_time_ms, tokens_per_second,
generation_ratio, non_generation_overhead_ms) is derived from
eval_duration/prompt_eval_duration/total_duration/eval_count, which are only
ever populated under timing_source == "ollama_runtime" -- see
token_features.py's module docstring for why mock leaves them None.

No attack/difficulty/split/branch/content field is read anywhere in this
module.
"""
from typing import Any, Dict, List, Optional

from ..models import AgentCallRecord

_NS_PER_MS = 1e6
_NS_PER_S = 1e9


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return a / b


def timing_features_for_session(agent_call_records: List[AgentCallRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for call in agent_call_records:
        generation_time_ms = call.eval_duration / _NS_PER_MS if call.eval_duration is not None else None
        prompt_eval_time_ms = (
            call.prompt_eval_duration / _NS_PER_MS if call.prompt_eval_duration is not None else None
        )
        non_generation_overhead_ms = None
        if (call.total_duration is not None and call.eval_duration is not None
                and call.prompt_eval_duration is not None):
            non_generation_overhead_ms = (
                call.total_duration - call.eval_duration - call.prompt_eval_duration
            ) / _NS_PER_MS
        tokens_per_second = None
        if call.eval_count is not None and call.eval_duration:
            tokens_per_second = call.eval_count / (call.eval_duration / _NS_PER_S)
        rows.append({
            "call_id": call.call_id,
            "agent_id": call.agent_id,
            "wall_clock_latency_ms": call.wall_clock_latency_ms,
            "generation_time_ms": generation_time_ms,
            "prompt_eval_time_ms": prompt_eval_time_ms,
            "tokens_per_second": tokens_per_second,
            "generation_ratio": _safe_div(call.eval_duration, call.total_duration),
            "non_generation_overhead_ms": non_generation_overhead_ms,
        })
    return rows
