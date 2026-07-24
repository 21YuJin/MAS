"""
[Step 7B] Candidate Feature Generator orchestrator -- combines the
token/timing/graph/session feature groups into one deterministic per-session
output. Raw records (AgentCallRecord/InteractionEvent/Message/Artifact/Part)
are only ever READ here, never mutated.

Deliberately excluded from this function's output:
  - normalization_features (needs an externally-fit normal_statistics
    mapping -- see normalization_features.py's module docstring, Step 8+)
  - any attack/difficulty/split/branch/content field -- this function's
    parameters (AgentCallRecord/InteractionEvent/Message/Artifact/Part lists)
    structurally cannot carry any of those fields (see models.py); no
    TravelTask or TaskInstance object is ever passed in here.
"""
from typing import Any, Dict, List

from ..models import AgentCallRecord, Artifact, InteractionEvent, Message, Part
from .graph_features import graph_features_for_session
from .session_features import session_features_for_session
from .timing_features import timing_features_for_session
from .token_features import token_features_for_session


def generate_candidate_features_for_session(
        agent_call_records: List[AgentCallRecord], events: List[InteractionEvent],
        messages: List[Message], artifacts: List[Artifact], parts: List[Part]) -> Dict[str, Any]:
    return {
        "token_features": token_features_for_session(agent_call_records),
        "timing_features": timing_features_for_session(agent_call_records),
        "graph_features": graph_features_for_session(events, messages, artifacts),
        "session_features": session_features_for_session(agent_call_records, messages, artifacts, parts),
    }
