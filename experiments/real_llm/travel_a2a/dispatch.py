"""
[Step 4-6/4-7] Shared communication/artifact instrumentation -- used by BOTH
the deterministic mock runner (mock_runner.py) and the Ollama-backed runner
(ollama_runner.py), so every Message/InteractionEvent this project produces,
regardless of which runner created it, goes through exactly one code path.
Neither runner hand-rolls its own status-transition bookkeeping or event
construction anymore (Step 3's mock_runner.py did this inline; Step 4
extracts it here so the Ollama runner doesn't have to duplicate it).

apply_action_result() does NOT decide what to do (that's still
workflow_policy.TravelWorkflowPolicy.decide()) and does NOT talk to any
agent (that's still mock_agents.py / ollama_agents.py) -- it only takes
whatever action+result already happened and turns it into the
Message/InteractionEvent/status-transition bookkeeping, uniformly.

create_artifact() enforces "never overwrite, always version up": a revision
is always a NEW Artifact object with version = prior.version + 1 and
parent_artifact_ids referencing the one it replaces -- callers never mutate
an existing Artifact's fields in place.
"""
import dataclasses
from typing import List, Optional

from .models import Artifact, ArtifactType, InteractionEvent, Message, Part, TravelTask
from .status import validate_status_transition
from .workflow_policy import INTERNAL_ACTION_TYPES, WorkflowAction


def create_artifact(id_factory, task: TravelTask, artifact_type: ArtifactType, producer_id: str,
                     part_ids: List[str], record_count: int, created_at: str,
                     prior_artifact: Optional[Artifact] = None,
                     source_artifact_ids: Optional[List[str]] = None,
                     parent_artifact_ids: Optional[List[str]] = None) -> Artifact:
    """
    Two distinct lineage cases, both handled here:
      - prior_artifact given (same artifact_type, e.g. hotel_options v1->v2):
        version = prior.version + 1, parent_artifact_ids = [prior.artifact_id]
        -- prior_artifact itself is never mutated, only referenced.
      - parent_artifact_ids given explicitly (cross-type lineage, e.g.
        final_travel_plan's parent is integrated_itinerary at version 1, not
        a revision of itself): version stays 1, parent_artifact_ids as given.
    Neither case overwrites an existing object -- every call returns a brand
    new Artifact.
    """
    if prior_artifact is not None:
        version = prior_artifact.version + 1
        parent_ids = [prior_artifact.artifact_id]
    else:
        version = 1
        parent_ids = list(parent_artifact_ids or [])
    return Artifact(
        artifact_id=id_factory.artifact_id(), task_id=task.task_id, context_id=task.context_id,
        artifact_type=artifact_type, producer_id=producer_id, version=version,
        parent_artifact_ids=parent_ids, source_artifact_ids=(source_artifact_ids or []),
        part_ids=part_ids, record_count=record_count, created_at=created_at, updated_at=created_at,
    )


@dataclasses.dataclass
class DispatchOutcome:
    status_before: object
    status_after: object
    status_transition_valid: bool
    input_part_ids: List[str]
    event: Optional[InteractionEvent]


def apply_action_result(action: WorkflowAction, result, task: TravelTask,
                         messages: List[Message], parts: List[Part], artifacts: List[Artifact],
                         events: List[InteractionEvent], session_id: str, id_factory,
                         start_ts: str, end_ts: str, prev_end_ts: Optional[str],
                         timing_source: str, status_transition_issues: list,
                         llm_called: bool = False, model_name: Optional[str] = None) -> DispatchOutcome:
    """
    Mutates messages/parts/artifacts/events in place (extends lists) and
    task.status, then returns a DispatchOutcome describing what happened --
    same behavior mock_runner.py's main loop used to inline directly.

    `result` is an AgentActionResult (mock_agents.py / ollama_agents.py) or
    None for a pure status-flip action (workflow_policy.NO_AGENT_CALL_ACTION_TYPES-
    style actions handled entirely by the caller before this is invoked).
    `llm_called`/`model_name` are informational passthrough onto the
    InteractionEvent for at-a-glance skimming -- the AUTHORITATIVE telemetry
    always lives on the AgentCallRecord the caller builds separately (Step 4-1);
    InteractionEvent.raw_ollama_telemetry is always {} here, never populated,
    regardless of runner.
    """
    if result is not None:
        parts.extend(result.generated_parts)
        artifacts.extend(result.generated_artifacts)

    status_before = task.status
    status_transition_valid = True
    if action.next_status is not None and action.next_status != status_before:
        status_transition_valid = validate_status_transition(status_before, action.next_status, mode="diagnostic")
        if not status_transition_valid:
            status_transition_issues.append({"event_index": len(events), "before": status_before.value,
                                              "after": action.next_status.value})
        task.status = action.next_status
    status_after = task.status

    input_part_ids = []
    request_message_id = action.context.get("request_message_id")
    if request_message_id:
        answered = next((m for m in messages if m.message_id == request_message_id), None)
        if answered is not None:
            input_part_ids = list(answered.part_ids)

    if action.action_type in INTERNAL_ACTION_TYPES:
        return DispatchOutcome(status_before, status_after, status_transition_valid, input_part_ids, None)

    generated_messages = result.generated_messages if result is not None else []
    generated_parts = result.generated_parts if result is not None else []
    generated_artifacts = result.generated_artifacts if result is not None else []
    messages.extend(generated_messages)
    message = generated_messages[0] if generated_messages else None

    event = InteractionEvent(
        event_id=id_factory.event_id(), event_index=len(events), session_id=session_id,
        task_id=task.task_id, context_id=task.context_id,
        sender_id=action.sender_id, receiver_id=action.receiver_id, interaction_type=action.interaction_type,
        message_id=(message.message_id if message else None),
        part_ids=[p.part_id for p in generated_parts], artifact_ids=[a.artifact_id for a in generated_artifacts],
        status_before=status_before, status_after=status_after, status_transition_valid=status_transition_valid,
        start_timestamp=start_ts, end_timestamp=end_ts, previous_event_timestamp=prev_end_ts,
        llm_called=llm_called, model_name=model_name, retry_count=0, error_flag=False, done_reason=None,
        raw_ollama_telemetry={}, timing_source=timing_source,
    )
    events.append(event)
    return DispatchOutcome(status_before, status_after, status_transition_valid, input_part_ids, event)
