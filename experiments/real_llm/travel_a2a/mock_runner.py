"""
[Step 3-5/3-7/3-8] Deterministic mock workflow driver + MockTravelSessionRunner.

run_mock_workflow() is the actual driver loop:

    while True:
        action = policy.decide(task, artifacts, messages, events, parts)
        if action is None:
            break
        execute action -> extend messages/parts/artifacts/events, update task.status

This is intentionally NOT:

    for sender, receiver in FIXED_EDGES:
        execute(sender, receiver)

-- every iteration re-asks the policy what's next, based on the state
accumulated so far (see workflow_policy.py's module docstring). The loop here
only executes whatever decide() already decided; it contains no branch logic
of its own about budget/schedule/missing-info/integration conditions.

Timing is produced by DeterministicClock -- fixed start point, fixed
increments -- so re-running the SAME task through run_mock_workflow() twice
(fresh id_factory + fresh clock both times) produces byte-identical output.
Every InteractionEvent this file creates is stamped timing_source=
"deterministic_mock" (Step 3-7) specifically so it can never be mistaken for
real Ollama telemetry once mixed into a shared dataset.
"""
import dataclasses
import datetime as dt
import os
import sys
from typing import List, Optional, Tuple

from .ids import DeterministicIdFactory
from .content_repository import ContentRepository, load_content_repository
from .dispatch import apply_action_result
from .mock_agents import build_mock_agent_registry
from .models import AgentCallRecord, Artifact, InteractionEvent, Message, Part, TravelTask
from .workflow_policy import TravelWorkflowPolicy

# [Step 3-8] runtime/ is a sibling package under experiments/real_llm/, one
# level up from travel_a2a/ -- see runtime/session_runner.py (Step 1-4).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runtime.session_runner import SessionRunner, SessionRunResult  # noqa: E402

MAX_STEPS = 200   # safety cap -- a real rule bug should raise, never spin forever

# Action types where sender_id == receiver_id == "travel_coordinator" but no
# agent call is even needed (pure status bookkeeping) -- distinct from
# "integrate_itinerary" (also internal, no Message/Event, but DOES need an
# agent call to actually produce the integrated_itinerary/final_travel_plan
# artifacts). See workflow_policy.INTERNAL_ACTION_TYPES for the broader set.
NO_AGENT_CALL_ACTION_TYPES = frozenset({"move_to_integrating", "resume_planning"})


class DeterministicClock:
    """Fixed-increment mock clock -- NOT wall-clock time. gap_ms elapses
    between the end of one event and the start of the next (dispatch/
    queueing overhead surrogate); step_ms is each mock "processing" step's
    own duration. Same (start, step_ms, gap_ms) always produces the same
    timestamp sequence for the same number of calls."""

    def __init__(self, start: str = "2026-01-01T00:00:00+00:00", step_ms: int = 20, gap_ms: int = 100):
        self._current = dt.datetime.fromisoformat(start)
        self._step_ms = step_ms
        self._gap_ms = gap_ms
        self._last_end: Optional[dt.datetime] = None

    def next_window(self) -> Tuple[str, str, Optional[str]]:
        if self._last_end is not None:
            self._current = self._last_end + dt.timedelta(milliseconds=self._gap_ms)
        start = self._current
        end = start + dt.timedelta(milliseconds=self._step_ms)
        previous_end = self._last_end
        self._last_end = end
        self._current = end
        return start.isoformat(), end.isoformat(), (previous_end.isoformat() if previous_end else None)


@dataclasses.dataclass
class MockWorkflowResult:
    task: TravelTask
    messages: List[Message]
    parts: List[Part]
    artifacts: List[Artifact]
    events: List[InteractionEvent]
    agent_call_records: List[AgentCallRecord]
    status_transition_issues: List[dict]


def run_mock_workflow(task: TravelTask, content_repository: ContentRepository,
                       id_factory: Optional[DeterministicIdFactory] = None,
                       clock: Optional[DeterministicClock] = None,
                       policy: Optional[TravelWorkflowPolicy] = None,
                       session_id: Optional[str] = None) -> MockWorkflowResult:
    id_factory = id_factory or DeterministicIdFactory()
    clock = clock or DeterministicClock()
    policy = policy or TravelWorkflowPolicy()
    session_id = session_id or f"session_{task.provenance.get('task_fixture_id', task.task_id)}"

    mock_agents = build_mock_agent_registry(content_repository)
    messages: List[Message] = []
    parts: List[Part] = []
    artifacts: List[Artifact] = []
    events: List[InteractionEvent] = []
    agent_call_records: List[AgentCallRecord] = []
    status_transition_issues: List[dict] = []

    for _ in range(MAX_STEPS + 1):
        action = policy.decide(task, artifacts, messages, events, parts=parts)
        if action is None:
            break

        start_ts, end_ts, prev_end_ts = clock.next_window()
        sequence_index = len(messages)

        if action.action_type in NO_AGENT_CALL_ACTION_TYPES:
            # Pure status flip -- no agent produces anything for this step
            # ("move_to_integrating" only exists to move task.status to
            # INTEGRATING before the coordinator actually integrates;
            # "resume_planning" only exists to satisfy status.py's
            # WAITING_FOR_INPUT -> PLANNING -> SEARCHING transition path).
            result = None
        else:
            agent = mock_agents[action.sender_id]
            result = agent.handle(action, task, artifacts, parts, id_factory, start_ts, sequence_index)

        outcome = apply_action_result(
            action, result, task, messages, parts, artifacts, events, session_id, id_factory,
            start_ts, end_ts, prev_end_ts, timing_source="deterministic_mock",
            status_transition_issues=status_transition_issues, llm_called=False, model_name=None)

        wall_clock_latency_ms = None
        if start_ts and end_ts:
            wall_clock_latency_ms = (dt.datetime.fromisoformat(end_ts) - dt.datetime.fromisoformat(start_ts)).total_seconds() * 1000.0

        agent_call_records.append(AgentCallRecord(
            call_id=id_factory.call_id(), session_id=session_id, task_id=task.task_id, context_id=task.context_id,
            agent_id=action.sender_id, action_type=action.action_type,
            triggering_message_id=action.context.get("request_message_id"),
            input_part_ids=outcome.input_part_ids,
            output_part_ids=([p.part_id for p in result.generated_parts] if result else []),
            output_artifact_ids=([a.artifact_id for a in result.generated_artifacts] if result else []),
            call_start_timestamp=start_ts, call_end_timestamp=end_ts, wall_clock_latency_ms=wall_clock_latency_ms,
            llm_called=False, model_name=None, retry_count=0, error_flag=False,
            timing_source="deterministic_mock",
        ))
    else:
        raise RuntimeError(f"workflow policy did not terminate within {MAX_STEPS} steps "
                            f"for task {task.task_id!r} -- likely rule bug (non-terminating loop)")

    return MockWorkflowResult(task=task, messages=messages, parts=parts, artifacts=artifacts, events=events,
                               agent_call_records=agent_call_records, status_transition_issues=status_transition_issues)


# ══════════════════════════════════════════════════════════════════════════
# [Step 3-8] MockTravelSessionRunner -- uses the Step 1 SessionRunner boundary
# ══════════════════════════════════════════════════════════════════════════


class MockTravelSessionRunner(SessionRunner):
    """Wraps run_mock_workflow() behind the Step 1 SessionRunner interface.
    Does not touch/modify LegacyRunSessionAdapter (real_llm's 4-agent
    pipeline) -- a separate, independent SessionRunner implementation for the
    travel_a2a mock workflow."""

    def __init__(self, content_repository: Optional[ContentRepository] = None):
        self.content_repository = content_repository or load_content_repository()

    def run(self, task: TravelTask, condition: str, attack_config: Optional[dict] = None, **kwargs) -> SessionRunResult:
        if condition != "normal":
            raise NotImplementedError("MockTravelSessionRunner: attack payloads are out of scope for Step 3 "
                                       "(see Step 3's 'do not implement' list)")
        result = run_mock_workflow(task, self.content_repository,
                                    id_factory=kwargs.get("id_factory"), clock=kwargs.get("clock"),
                                    session_id=kwargs.get("session_id"))

        final_plan = next((a for a in result.artifacts if a.artifact_type.value == "final_travel_plan"), None)
        final_output = None
        if final_plan is not None:
            part = next((p for p in result.parts if p.part_id in final_plan.part_ids), None)
            final_output = str(part.content) if part is not None else None

        return SessionRunResult(
            session_id=kwargs.get("session_id"), task_id=task.task_id, context_id=task.context_id,
            agent_call_records=[r.to_dict() for r in result.agent_call_records],
            messages=[m.to_dict() for m in result.messages],
            parts=[p.to_dict() for p in result.parts],
            artifacts=[a.to_dict() for a in result.artifacts],
            interaction_events=[e.to_dict() for e in result.events],
            final_output=final_output,
            diagnostic_labels={"task_status": task.status.value, "status_transition_issues": result.status_transition_issues},
            errors=[],
        )
