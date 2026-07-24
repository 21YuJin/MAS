"""
[Step 7B] Graph Feature Group -- derived from InteractionEvent (edge-level),
Message.request_message_id (reply-chain), and Artifact.parent_artifact_ids
(lineage). All fully populated under deterministic_mock -- unlike
token_features/timing_features, this group needs no Ollama telemetry and is
exercisable end-to-end today (Step 6.5D already validated this via
event_pattern_signature/graph_pattern_signature diversity).

No attack/difficulty/split/branch/content field is read anywhere in this
module.
"""
from collections import defaultdict
from typing import Any, Dict, List

from ..models import Artifact, InteractionEvent, Message


def fan_in_out_by_agent(events: List[InteractionEvent]) -> Dict[str, Dict[str, int]]:
    fan_in: Dict[str, int] = defaultdict(int)
    fan_out: Dict[str, int] = defaultdict(int)
    for e in events:
        fan_out[e.sender_id] += 1
        fan_in[e.receiver_id] += 1
    agents = set(fan_in) | set(fan_out)
    return {a: {"fan_in": fan_in.get(a, 0), "fan_out": fan_out.get(a, 0)} for a in agents}


def edge_density(events: List[InteractionEvent]) -> float:
    agents = {e.sender_id for e in events} | {e.receiver_id for e in events}
    n = len(agents)
    if n < 2:
        return 0.0
    unique_pairs = {(e.sender_id, e.receiver_id) for e in events}
    return len(unique_pairs) / (n * (n - 1))


def parallel_edge_count(events: List[InteractionEvent]) -> int:
    """Count of directed (sender, receiver) pairs seen more than once in this
    session. This graph is temporal/sequential (one session, one active
    workflow), not concurrent -- 'parallel' here means repeated-over-time,
    the same convention formal_workload_mock_run.py's repeated_pair_count
    already uses, not simultaneous execution."""
    pair_counts: Dict[tuple, int] = defaultdict(int)
    for e in events:
        pair_counts[(e.sender_id, e.receiver_id)] += 1
    return sum(1 for _, n in pair_counts.items() if n > 1)


def reply_chain_depth(messages: List[Message]) -> int:
    """Longest request_message_id reply chain -- this environment's Message
    model has no explicit DAG-depth field, so this is the closest available
    analogue to the spec's 'path_depth'."""
    by_id = {m.message_id: m for m in messages}
    depth_cache: Dict[str, int] = {}

    def depth_of(message_id: str) -> int:
        if message_id in depth_cache:
            return depth_cache[message_id]
        m = by_id.get(message_id)
        if m is None or m.request_message_id is None or m.request_message_id not in by_id:
            depth_cache[message_id] = 1
        else:
            depth_cache[message_id] = 1 + depth_of(m.request_message_id)
        return depth_cache[message_id]

    return max((depth_of(m.message_id) for m in messages), default=0)


def artifact_lineage_depth(artifacts: List[Artifact]) -> int:
    by_id = {a.artifact_id: a for a in artifacts}
    depth_cache: Dict[str, int] = {}

    def depth_of(artifact_id: str) -> int:
        if artifact_id in depth_cache:
            return depth_cache[artifact_id]
        a = by_id.get(artifact_id)
        parents = [p for p in a.parent_artifact_ids if p in by_id] if a is not None else []
        depth_cache[artifact_id] = 1 if not parents else 1 + max(depth_of(p) for p in parents)
        return depth_cache[artifact_id]

    return max((depth_of(a.artifact_id) for a in artifacts), default=0)


def graph_features_for_session(events: List[InteractionEvent], messages: List[Message],
                                artifacts: List[Artifact]) -> Dict[str, Any]:
    return {
        "fan_in_out_by_agent": fan_in_out_by_agent(events),
        "edge_density": edge_density(events),
        "parallel_edge_count": parallel_edge_count(events),
        "reply_chain_depth": reply_chain_depth(messages),
        "artifact_lineage_depth": artifact_lineage_depth(artifacts),
    }
