"""
[Step 4] Ollama-backed workflow driver. Reuses workflow_policy.py's
TravelWorkflowPolicy.decide() completely UNCHANGED from mock_runner.py --
the routing/branch decisions are identical; the only things that differ are
(a) which agent registry executes each action (real Ollama calls via
ollama_agents.py vs. deterministic mocks via mock_agents.py) and (b) that
timestamps are real wall-clock time (Ollama calls take real, variable time),
not DeterministicClock's fixed increments.

Per the Step 4-5 instruction, the LLM never picks the next agent or decides
whether a revision/clarification is needed -- that's still entirely
workflow_policy.py's job, identical to the mock run.
"""
import datetime as dt
import os
import sys
from typing import List, Optional

from .attack_models import AttackConfig
from .content_repository import ContentRepository, load_content_repository
from .dispatch import apply_action_result
from .ids import DeterministicIdFactory
from .mock_runner import MAX_STEPS, MockWorkflowResult, NO_AGENT_CALL_ACTION_TYPES
from .models import AgentCallRecord
from .ollama_agents import build_ollama_agent_registry
from .workflow_policy import TravelWorkflowPolicy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runtime.session_runner import SessionRunner, SessionRunResult  # noqa: E402


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_ollama_workflow(task, content_repository: ContentRepository,
                         id_factory: Optional[DeterministicIdFactory] = None,
                         policy: Optional[TravelWorkflowPolicy] = None,
                         session_id: Optional[str] = None,
                         attack_config: Optional[AttackConfig] = None) -> MockWorkflowResult:
    id_factory = id_factory or DeterministicIdFactory()
    policy = policy or TravelWorkflowPolicy()
    session_id = session_id or f"session_{task.provenance.get('task_fixture_id', task.task_id)}"

    agents = build_ollama_agent_registry(content_repository)
    messages, parts, artifacts, events, agent_call_records = [], [], [], [], []
    status_transition_issues: List[dict] = []
    prev_end_ts: Optional[str] = None

    for _ in range(MAX_STEPS + 1):
        action = policy.decide(task, artifacts, messages, events, parts=parts)
        if action is None:
            break

        start_ts = _now_iso()
        sequence_index = len(messages)

        if action.action_type in NO_AGENT_CALL_ACTION_TYPES:
            result = None
        else:
            agent = agents[action.sender_id]
            # attack_config is passed to EVERY agent call -- each agent's own
            # handle() decides whether it applies (only the one whose own
            # role matches attack_config.entry_agent_id actually injects
            # anything, see injection_builder.apply_attack_injection()).
            result = agent.handle(action, task, artifacts, parts, id_factory, start_ts, sequence_index,
                                   session_id=session_id, attack_config=attack_config)

        end_ts = _now_iso()
        has_call_record = result is not None and result.call_record is not None
        llm_called = bool(result.call_record.llm_called) if has_call_record else False
        model_name = result.call_record.model_name if has_call_record else None

        outcome = apply_action_result(
            action, result, task, messages, parts, artifacts, events, session_id, id_factory,
            start_ts, end_ts, prev_end_ts, timing_source="ollama_runtime",
            status_transition_issues=status_transition_issues, llm_called=llm_called, model_name=model_name)
        prev_end_ts = end_ts

        if has_call_record:
            agent_call_records.append(result.call_record)
        else:
            wall_clock_latency_ms = (dt.datetime.fromisoformat(end_ts) - dt.datetime.fromisoformat(start_ts)).total_seconds() * 1000.0
            agent_call_records.append(AgentCallRecord(
                call_id=id_factory.call_id(), session_id=session_id, task_id=task.task_id, context_id=task.context_id,
                agent_id=action.sender_id, action_type=action.action_type,
                triggering_message_id=action.context.get("request_message_id"),
                input_part_ids=outcome.input_part_ids,
                output_part_ids=([p.part_id for p in result.generated_parts] if result else []),
                output_artifact_ids=([a.artifact_id for a in result.generated_artifacts] if result else []),
                call_start_timestamp=start_ts, call_end_timestamp=end_ts, wall_clock_latency_ms=wall_clock_latency_ms,
                llm_called=False, model_name=None, retry_count=0, error_flag=False, timing_source="ollama_runtime",
            ))
    else:
        raise RuntimeError(f"workflow policy did not terminate within {MAX_STEPS} steps "
                            f"for task {task.task_id!r} -- likely rule bug (non-terminating loop)")

    return MockWorkflowResult(task=task, messages=messages, parts=parts, artifacts=artifacts, events=events,
                               agent_call_records=agent_call_records, status_transition_issues=status_transition_issues)


class OllamaTravelSessionRunner(SessionRunner):
    """Same SessionRunner interface as MockTravelSessionRunner -- swap this
    in wherever that one was used to get real Ollama telemetry instead of
    deterministic mock output, with no other code changes required."""

    def __init__(self, content_repository: Optional[ContentRepository] = None):
        self.content_repository = content_repository or load_content_repository()

    def run(self, task, condition: str, attack_config: Optional[AttackConfig] = None, **kwargs) -> SessionRunResult:
        if condition == "attack" and attack_config is None:
            raise ValueError("condition='attack' requires attack_config (Step 5)")
        if condition == "normal" and attack_config is not None:
            raise ValueError("attack_config must be None when condition='normal'")
        result = run_ollama_workflow(task, self.content_repository, id_factory=kwargs.get("id_factory"),
                                      session_id=kwargs.get("session_id"), attack_config=attack_config)

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
