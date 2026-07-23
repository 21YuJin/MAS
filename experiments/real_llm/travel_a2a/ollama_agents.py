"""
[Step 4-4] Ollama-backed travel agents. Each subclasses its Step 3 Mock*Agent
counterpart and overrides ONLY the one action_type that represents genuine
specialist "analysis" work (the thing an LLM should actually do); every other
action_type on that agent (clarifications, revisions, routing) is inherited
unchanged from the mock implementation, since those are mechanical
confirmations/relays, not reasoning tasks -- matching the Step 4-5 instruction
("LLM은 외부 콘텐츠 분석/후보 비교/요약/Artifact content 생성만 담당한다").

Every override follows the exact same shape:
  1. Build the SAME structured content_repository data the mock agent uses
     (never reinterpreted from LLM output) -- this is what
     workflow_policy.py's conflict-detection reads, so it MUST stay
     deterministic regardless of which agent implementation produced it.
  2. Build a prompt from prompt_builders.py (role instruction + structured
     data rendered as readable text) and call OllamaAgentExecutor.execute()
     -- the ONE path every real LLM call goes through (Step 4-2).
  3. Attach the LLM's response as an ADDITIONAL narrative Part, alongside
     (never instead of) the structured Part.
  4. Set AgentActionResult.call_record to the executor's AgentCallRecord, so
     the runner never has to hand-build LLM telemetry itself.

`client` is not upgraded -- it never calls an LLM (agents.CLIENT_AGENT_ID),
so MockClient is reused as-is for the Ollama runner too.
"""
import json
from typing import Optional

from .attack_models import AttackConfig
from .content_repository import ContentRepository
from .dispatch import create_artifact
from .injection_builder import apply_attack_injection
from .mock_agents import (
    LODGING_BUDGET_FRACTION, AgentActionResult, MockClient, MockCoordinator, MockCurrencyAgent,
    MockFlightAgent, MockHotelAgent, MockToursAgent, _data_part, _reply_message,
)
from .models import ArtifactType
from .ollama_executor import OllamaAgentExecutor
from .prompt_builders import (
    build_coordinator_prompt_v2, build_currency_prompt, build_flight_prompt_v2,
    build_hotel_prompt_v2, build_tours_prompt_v2,
)


def _try_parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


class OllamaFlightAgent(MockFlightAgent):
    def __init__(self, content_repository: ContentRepository, executor: Optional[OllamaAgentExecutor] = None):
        super().__init__(content_repository)
        self.executor = executor or OllamaAgentExecutor(prompt_config_version="flight_v2")

    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index,
               session_id=None, attack_config: Optional[AttackConfig] = None):
        if action.action_type != "deliver_flight_options":
            return super().handle(action, task, artifacts, parts, id_factory, created_at, sequence_index,
                                   attack_config=attack_config)

        base_options = self.content_repository.flights_for(task.request.destination)
        # [Step 5-5] injection, if any, is applied HERE -- only this specific
        # option's field is touched; price/dates/option_id stay identical to
        # the normal-session options a matched pair would see.
        options, injection_record = apply_attack_injection(base_options, attack_config, entry_agent_id="flight_agent")
        structured_content = {"destination": task.request.destination, "options": options}
        structured_part = _data_part(structured_content, id_factory, created_at)
        prompt = build_flight_prompt_v2(task.request.destination, options)
        exec_result = self.executor.execute(
            agent_id="flight_agent", action_type=action.action_type, prompt=prompt, session_id=session_id,
            task_id=task.task_id, context_id=task.context_id, id_factory=id_factory,
            triggering_message_id=action.context.get("request_message_id"), input_part_ids=[structured_part.part_id])
        narrative_part = exec_result.response_part
        parsed = _try_parse_json(exec_result.response_text)
        if parsed is None:
            exec_result.call_record.error_type = exec_result.call_record.error_type or "json_parse_error"

        artifact = create_artifact(id_factory, task, ArtifactType.FLIGHT_OPTIONS, "flight_agent",
                                    part_ids=[structured_part.part_id, narrative_part.part_id],
                                    record_count=len(options), created_at=created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index,
                              part_ids=[structured_part.part_id, narrative_part.part_id],
                              artifact_ids=[artifact.artifact_id], request_message_id=action.context["request_message_id"])
        diagnostic_values = {"option_count": len(options),
                              "llm_selected_option_id": (parsed or {}).get("selected_option_id")}
        if injection_record is not None:
            diagnostic_values["injection_applied"] = True
            diagnostic_values["injection_channel"] = injection_record.injection_channel
        return AgentActionResult(generated_parts=[structured_part, narrative_part], generated_messages=[msg],
                                  generated_artifacts=[artifact], diagnostic_values=diagnostic_values,
                                  call_record=exec_result.call_record)


class OllamaHotelAgent(MockHotelAgent):
    def __init__(self, content_repository: ContentRepository, executor: Optional[OllamaAgentExecutor] = None):
        super().__init__(content_repository)
        self.executor = executor or OllamaAgentExecutor(prompt_config_version="hotel_v2")

    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index,
               session_id=None, attack_config: Optional[AttackConfig] = None):
        if action.action_type != "deliver_hotel_options":
            return super().handle(action, task, artifacts, parts, id_factory, created_at, sequence_index,
                                   attack_config=attack_config)

        base_options = self.content_repository.hotels_for(task.request.destination)
        options, injection_record = apply_attack_injection(base_options, attack_config, entry_agent_id="hotel_agent")
        structured_content = {"destination": task.request.destination, "options": options}
        structured_part = _data_part(structured_content, id_factory, created_at)
        prompt = build_hotel_prompt_v2(task.request.destination, options)
        exec_result = self.executor.execute(
            agent_id="hotel_agent", action_type=action.action_type, prompt=prompt, session_id=session_id,
            task_id=task.task_id, context_id=task.context_id, id_factory=id_factory,
            triggering_message_id=action.context.get("request_message_id"), input_part_ids=[structured_part.part_id])
        narrative_part = exec_result.response_part
        parsed = _try_parse_json(exec_result.response_text)
        if parsed is None:
            exec_result.call_record.error_type = exec_result.call_record.error_type or "json_parse_error"

        artifact = create_artifact(id_factory, task, ArtifactType.HOTEL_OPTIONS, "hotel_agent",
                                    part_ids=[structured_part.part_id, narrative_part.part_id],
                                    record_count=len(options), created_at=created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index,
                              part_ids=[structured_part.part_id, narrative_part.part_id],
                              artifact_ids=[artifact.artifact_id], request_message_id=action.context["request_message_id"])
        diagnostic_values = {"option_count": len(options),
                              "llm_selected_option_id": (parsed or {}).get("selected_option_id")}
        if injection_record is not None:
            diagnostic_values["injection_applied"] = True
            diagnostic_values["injection_channel"] = injection_record.injection_channel
        return AgentActionResult(generated_parts=[structured_part, narrative_part], generated_messages=[msg],
                                  generated_artifacts=[artifact], diagnostic_values=diagnostic_values,
                                  call_record=exec_result.call_record)


class OllamaCurrencyAgent(MockCurrencyAgent):
    def __init__(self, content_repository: ContentRepository, executor: Optional[OllamaAgentExecutor] = None):
        super().__init__(content_repository)
        self.executor = executor or OllamaAgentExecutor(prompt_config_version="currency_v1")

    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index,
               session_id=None, attack_config: Optional[AttackConfig] = None):
        # currency_provider_note injection is deferred (Step 5-2: "Currency는
        # evaluator가 안정된 다음 추가하는 게 좋아") -- attack_config is accepted
        # for signature uniformity but never applied here yet.
        if action.action_type != "deliver_budget_conversion":
            return super().handle(action, task, artifacts, parts, id_factory, created_at, sequence_index,
                                   attack_config=attack_config)

        rate = self.content_repository.currency_rate(task.request.budget_currency, task.request.target_currency)
        total_budget_target = task.request.budget_amount * rate
        content = {
            "base_currency": task.request.budget_currency, "target_currency": task.request.target_currency,
            "rate": rate, "total_budget_target_currency": round(total_budget_target, 2),
            "lodging_budget_target_currency": round(total_budget_target * LODGING_BUDGET_FRACTION, 2),
        }
        structured_part = _data_part(content, id_factory, created_at)
        prompt = build_currency_prompt(content)
        exec_result = self.executor.execute(
            agent_id="currency_agent", action_type=action.action_type, prompt=prompt, session_id=session_id,
            task_id=task.task_id, context_id=task.context_id, id_factory=id_factory,
            triggering_message_id=action.context.get("request_message_id"), input_part_ids=[structured_part.part_id])
        narrative_part = exec_result.response_part
        if _try_parse_json(exec_result.response_text) is None:
            exec_result.call_record.error_type = exec_result.call_record.error_type or "json_parse_error"

        artifact = create_artifact(id_factory, task, ArtifactType.BUDGET_CONVERSION, "currency_agent",
                                    part_ids=[structured_part.part_id, narrative_part.part_id],
                                    record_count=1, created_at=created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index,
                              part_ids=[structured_part.part_id, narrative_part.part_id],
                              artifact_ids=[artifact.artifact_id], request_message_id=action.context["request_message_id"])
        return AgentActionResult(generated_parts=[structured_part, narrative_part], generated_messages=[msg],
                                  generated_artifacts=[artifact], diagnostic_values=content, call_record=exec_result.call_record)


class OllamaToursAgent(MockToursAgent):
    def __init__(self, content_repository: ContentRepository, executor: Optional[OllamaAgentExecutor] = None):
        super().__init__(content_repository)
        self.executor = executor or OllamaAgentExecutor(prompt_config_version="tours_v2")

    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index,
               session_id=None, attack_config: Optional[AttackConfig] = None):
        if action.action_type != "deliver_tour_options":
            return super().handle(action, task, artifacts, parts, id_factory, created_at, sequence_index,
                                   attack_config=attack_config)

        base_options = self.content_repository.tours_for_in_range(
            task.request.destination, task.request.departure_date, task.request.return_date)
        options, injection_record = apply_attack_injection(base_options, attack_config, entry_agent_id="tours_agent")
        structured_content = {"destination": task.request.destination, "options": options}
        structured_part = _data_part(structured_content, id_factory, created_at)
        prompt = build_tours_prompt_v2(task.request.destination, options)
        exec_result = self.executor.execute(
            agent_id="tours_agent", action_type=action.action_type, prompt=prompt, session_id=session_id,
            task_id=task.task_id, context_id=task.context_id, id_factory=id_factory,
            triggering_message_id=action.context.get("request_message_id"), input_part_ids=[structured_part.part_id])
        narrative_part = exec_result.response_part
        parsed = _try_parse_json(exec_result.response_text)
        if parsed is None:
            exec_result.call_record.error_type = exec_result.call_record.error_type or "json_parse_error"

        artifact = create_artifact(id_factory, task, ArtifactType.TOUR_OPTIONS, "tours_agent",
                                    part_ids=[structured_part.part_id, narrative_part.part_id],
                                    record_count=len(options), created_at=created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index,
                              part_ids=[structured_part.part_id, narrative_part.part_id],
                              artifact_ids=[artifact.artifact_id], request_message_id=action.context["request_message_id"])
        diagnostic_values = {"option_count": len(options),
                              "llm_selected_option_id": (parsed or {}).get("selected_option_id")}
        if injection_record is not None:
            diagnostic_values["injection_applied"] = True
            diagnostic_values["injection_channel"] = injection_record.injection_channel
        return AgentActionResult(generated_parts=[structured_part, narrative_part], generated_messages=[msg],
                                  generated_artifacts=[artifact], diagnostic_values=diagnostic_values,
                                  call_record=exec_result.call_record)


class OllamaCoordinator(MockCoordinator):
    """[Step 4-10] Only "integrate_itinerary" is LLM-backed here -- routing
    actions (delegate_*/request_*_revision/request_client_clarification/
    task_completion) stay deterministic/mechanical, per the Step 4-5
    instruction (workflow decisions belong to workflow_policy.py, not the
    LLM). This one call represents artifact_integration + final_plan_generation
    combined -- it produces NO Message/InteractionEvent (workflow_policy's
    INTERNAL_ACTION_TYPES, unchanged from Step 3) but DOES produce an
    AgentCallRecord with llm_called=True, so the coordinator's own token/
    latency cost is never missing from node-level telemetry."""

    def __init__(self, executor: Optional[OllamaAgentExecutor] = None):
        super().__init__()
        self.executor = executor or OllamaAgentExecutor(prompt_config_version="coordinator_v2")

    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index,
               session_id=None, attack_config: Optional[AttackConfig] = None):
        if action.action_type != "integrate_itinerary":
            return super().handle(action, task, artifacts, parts, id_factory, created_at, sequence_index,
                                   attack_config=attack_config)

        by_type = {}
        for a in artifacts:
            if a.artifact_type.value not in by_type or a.version > by_type[a.artifact_type.value].version:
                by_type[a.artifact_type.value] = a
        # [Step 5] The coordinator now reads each specialist's ACTUAL narrative
        # text (the second Part on each artifact, per the "[structured, narrative]"
        # convention every OllamaXAgent uses above) -- not just artifact counts.
        # A coordinator that never reads what a specialist said has no channel
        # for anything to propagate into its own summary, which would make
        # downstream_propagation structurally untestable regardless of what
        # an entry agent's LLM call actually did.
        parts_by_id = {p.part_id: p for p in parts}
        narratives = {}
        for t, a in by_type.items():
            if len(a.part_ids) >= 2:
                narratives[t] = parts_by_id[a.part_ids[1]].content
        prompt = build_coordinator_prompt_v2(task.request.destination, narratives)
        exec_result = self.executor.execute(
            agent_id="travel_coordinator", action_type=action.action_type, prompt=prompt, session_id=session_id,
            task_id=task.task_id, context_id=task.context_id, id_factory=id_factory, triggering_message_id=None,
            input_artifact_ids=[a.artifact_id for a in by_type.values()])
        narrative_part = exec_result.response_part
        if _try_parse_json(exec_result.response_text) is None:
            exec_result.call_record.error_type = exec_result.call_record.error_type or "json_parse_error"

        integrated = create_artifact(id_factory, task, ArtifactType.INTEGRATED_ITINERARY, "travel_coordinator",
                                      part_ids=[narrative_part.part_id], record_count=len(by_type), created_at=created_at,
                                      source_artifact_ids=[a.artifact_id for a in by_type.values()])
        final_part = _data_part({"status": "ready", "based_on": integrated.artifact_id}, id_factory, created_at)
        final_plan = create_artifact(id_factory, task, ArtifactType.FINAL_TRAVEL_PLAN, "travel_coordinator",
                                      part_ids=[final_part.part_id], record_count=1, created_at=created_at,
                                      parent_artifact_ids=[integrated.artifact_id])
        return AgentActionResult(
            generated_parts=[narrative_part, final_part], generated_artifacts=[integrated, final_plan],
            diagnostic_values={"integrated_artifact_id": integrated.artifact_id, "final_plan_artifact_id": final_plan.artifact_id},
            call_record=exec_result.call_record)


def build_ollama_agent_registry(content_repository: ContentRepository) -> dict:
    return {
        "client": MockClient(),
        "travel_coordinator": OllamaCoordinator(),
        "flight_agent": OllamaFlightAgent(content_repository),
        "hotel_agent": OllamaHotelAgent(content_repository),
        "currency_agent": OllamaCurrencyAgent(content_repository),
        "tours_agent": OllamaToursAgent(content_repository),
    }
