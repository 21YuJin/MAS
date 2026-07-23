"""
[Step 3-4] Deterministic mock agents -- produce Part/Message/Artifact objects
from fixed rules over the content_repository, standing in for an LLM agent
during workflow-structure development. The goal here is NOT convincing
generated text -- it's an exactly reproducible execution STRUCTURE (same
task + same fixtures => same artifact types/counts/branches/sender-receiver
sequence every time) that Step 4 can later swap for real Ollama calls one
agent at a time, without touching workflow_policy.py's decision logic.

Every mock agent exposes ONE entry point, `handle(action, ...)`, dispatching
internally on `action.action_type`. The convention used throughout this
package: the agent that ACTS for a given WorkflowAction is always
`action.sender_id` (true whether the action is an outbound request like a
task_delegation, or an outbound response like an artifact_delivery) -- so the
runner can always look up "who acts" via one rule, never a per-action_type
special case.

Business rules (uniform across every destination/task -- only the content
fixtures make some tasks hit a branch and others not):
  - lodging budget = LODGING_BUDGET_FRACTION of the task's total budget,
    converted to target_currency
  - "cheapest" flight/hotel selection = min(price) among delivered options
  - a hotel revision applies a flat BUDGET_REVISION_DISCOUNT to the cheapest
    option
  - a tours schedule conflict is resolved by dropping the conflicting
    option(s), never by changing the flight
"""
import dataclasses
from typing import Callable, List, Optional

from .content_repository import ContentRepository, content_record_to_part
from .models import (
    AgentCallRecord, Artifact, ArtifactType, InteractionType, Message, Part, PartType,
    SourceType, TravelTask,
)
from .workflow_policy import WorkflowAction

LODGING_BUDGET_FRACTION = 0.35
BUDGET_REVISION_DISCOUNT = 0.30


@dataclasses.dataclass
class AgentActionResult:
    generated_parts: List[Part] = dataclasses.field(default_factory=list)
    generated_messages: List[Message] = dataclasses.field(default_factory=list)
    generated_artifacts: List[Artifact] = dataclasses.field(default_factory=list)
    diagnostic_values: dict = dataclasses.field(default_factory=dict)
    next_action_hint: Optional[str] = None
    # [Step 4-1/4-2] Set by Ollama-backed agents (ollama_agents.py) to the
    # rich AgentCallRecord OllamaAgentExecutor already built (full telemetry).
    # Always None from every Mock*Agent -- mock_runner.py builds its own
    # (llm_called=False, no telemetry) AgentCallRecord directly, since there
    # is no executor call for it to come from.
    call_record: Optional[AgentCallRecord] = None


def _data_part(content, id_factory, created_at, source_type=SourceType.AGENT_GENERATED) -> Part:
    return Part(part_id=id_factory.part_id(), part_type=PartType.DATA, mime_type="application/json",
                content=content, source_type=source_type, created_at=created_at,
                injection_present=False, attack_id=None)


def _reply_message(action: WorkflowAction, id_factory, created_at, sequence_index,
                    part_ids=None, artifact_ids=None, request_message_id=None) -> Message:
    return Message(
        message_id=id_factory.message_id(), task_id=action.context["task_id"],
        context_id=action.context["context_id"], sender_id=action.sender_id,
        receiver_id=action.receiver_id, interaction_type=action.interaction_type,
        role=("client" if action.sender_id == "client" else "agent"),
        part_ids=(part_ids or []), artifact_ids=(artifact_ids or []),
        request_message_id=request_message_id, sequence_index=sequence_index,
        created_at=created_at,
    )


class MockClient:
    """Not one of the 5 LLM agents (client never calls an LLM, per
    agents.CLIENT_AGENT_ID) -- still needs to "act" in the mock workflow
    (submit the task, answer clarification requests), so it gets the same
    handle() entry point for a uniform dispatch table in mock_runner.py."""

    def handle(self, action, task: TravelTask, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        if action.action_type == "submit_task":
            part = _data_part(task.request.to_dict(), id_factory, created_at, source_type=SourceType.USER_REQUEST)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        if action.action_type == "client_clarification_response":
            # Deterministic canned answer -- fixed regardless of task, since
            # this is a structure test, not a content-quality test.
            answer = {"hotel_preferences": {"room_type": "standard", "location_preference": "city_center"}}
            part = _data_part(answer, id_factory, created_at, source_type=SourceType.USER_REQUEST)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                                  request_message_id=action.context["request_message_id"])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        raise ValueError(f"MockClient cannot handle action_type {action.action_type!r}")


class MockCoordinator:
    def handle(self, action, task: TravelTask, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        handler = {
            "request_client_clarification": self._request_client_clarification,
            "delegate_flight_search": self._delegate,
            "delegate_hotel_search": self._delegate,
            "delegate_currency_check": self._delegate,
            "delegate_tours_search": self._delegate,
            "request_hotel_revision": self._request_revision,
            "request_integration_revision": self._request_revision,
            "integrate_itinerary": self._integrate_itinerary,
            "task_completion": self._task_completion,
        }.get(action.action_type)
        if handler is None:
            raise ValueError(f"MockCoordinator cannot handle action_type {action.action_type!r}")
        return handler(action, task, artifacts, parts, id_factory, created_at, sequence_index)

    def _request_client_clarification(self, action, task, artifacts, parts, id_factory, created_at, sequence_index):
        part = _data_part({"missing_field": "hotel_preferences", "question": "What hotel room type / area do you prefer?"},
                           id_factory, created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id])
        return AgentActionResult(generated_parts=[part], generated_messages=[msg])

    def _delegate(self, action, task, artifacts, parts, id_factory, created_at, sequence_index):
        req = task.request
        payload_by_type = {
            "delegate_flight_search": {"origin": req.origin, "destination": req.destination,
                                        "departure_date": req.departure_date, "return_date": req.return_date,
                                        "flight_preferences": req.flight_preferences},
            "delegate_hotel_search": {"destination": req.destination, "check_in": req.departure_date,
                                       "check_out": req.return_date, "hotel_preferences": req.hotel_preferences},
            "delegate_currency_check": {"budget_amount": req.budget_amount, "budget_currency": req.budget_currency,
                                         "target_currency": req.target_currency},
            "delegate_tours_search": {"destination": req.destination, "departure_date": req.departure_date,
                                       "return_date": req.return_date, "activity_preferences": req.activity_preferences},
        }
        part = _data_part(payload_by_type[action.action_type], id_factory, created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id])
        return AgentActionResult(generated_parts=[part], generated_messages=[msg])

    def _request_revision(self, action, task, artifacts, parts, id_factory, created_at, sequence_index):
        part = _data_part(dict(action.context.get("reason_payload", {})), id_factory, created_at)
        msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                              artifact_ids=[action.context["target_artifact_id"]])
        return AgentActionResult(generated_parts=[part], generated_messages=[msg])

    def _integrate_itinerary(self, action, task, artifacts, parts, id_factory, created_at, sequence_index):
        """[Step 3-5, step 10] Internal artifact production -- deliberately no
        Message/InteractionEvent (see workflow_policy.py's docstring): the base
        flow lists this step without a sender->receiver arrow."""
        by_type = {}
        for a in artifacts:
            if a.artifact_type.value not in by_type or a.version > by_type[a.artifact_type.value].version:
                by_type[a.artifact_type.value] = a

        summary_parts_content = {t: {"artifact_id": a.artifact_id, "version": a.version, "record_count": a.record_count}
                                  for t, a in by_type.items()}
        integrated_part = _data_part({"summary": summary_parts_content}, id_factory, created_at)
        integrated = Artifact(
            artifact_id=id_factory.artifact_id(), task_id=task.task_id, context_id=task.context_id,
            artifact_type=ArtifactType.INTEGRATED_ITINERARY, producer_id="travel_coordinator",
            version=1, source_artifact_ids=[a.artifact_id for a in by_type.values()],
            part_ids=[integrated_part.part_id], record_count=len(by_type), created_at=created_at, updated_at=created_at,
        )
        final_part = _data_part({"status": "ready", "based_on": integrated.artifact_id}, id_factory, created_at)
        final_plan = Artifact(
            artifact_id=id_factory.artifact_id(), task_id=task.task_id, context_id=task.context_id,
            artifact_type=ArtifactType.FINAL_TRAVEL_PLAN, producer_id="travel_coordinator",
            version=1, parent_artifact_ids=[integrated.artifact_id],
            part_ids=[final_part.part_id], record_count=1, created_at=created_at, updated_at=created_at,
        )
        return AgentActionResult(
            generated_parts=[integrated_part, final_part],
            generated_artifacts=[integrated, final_plan],
            diagnostic_values={"integrated_artifact_id": integrated.artifact_id, "final_plan_artifact_id": final_plan.artifact_id},
        )

    def _task_completion(self, action, task, artifacts, parts, id_factory, created_at, sequence_index):
        final_plan = next(a for a in artifacts if a.artifact_type == ArtifactType.FINAL_TRAVEL_PLAN)
        msg = _reply_message(action, id_factory, created_at, sequence_index, artifact_ids=[final_plan.artifact_id])
        return AgentActionResult(generated_messages=[msg])


class _SpecialistBase:
    """Shared plumbing for the 4 specialist mock agents -- content lookup +
    Part/Artifact/Message construction. Subclasses only need to supply the
    per-action handler map and the artifact_type/content lookup logic."""

    def __init__(self, content_repository: ContentRepository):
        self.content_repository = content_repository

    def _deliver(self, action, task, artifacts, id_factory, created_at, sequence_index,
                 artifact_type: ArtifactType, options: list, version: int = 1,
                 parent_artifact_ids=None, request_message_id=None) -> AgentActionResult:
        content = {"destination": task.request.destination, "options": options}
        part = _data_part(content, id_factory, created_at)
        artifact = Artifact(
            artifact_id=id_factory.artifact_id(), task_id=task.task_id, context_id=task.context_id,
            artifact_type=artifact_type, producer_id=action.sender_id, version=version,
            parent_artifact_ids=(parent_artifact_ids or []), part_ids=[part.part_id],
            record_count=len(options), created_at=created_at, updated_at=created_at,
        )
        msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                              artifact_ids=[artifact.artifact_id], request_message_id=request_message_id)
        return AgentActionResult(generated_parts=[part], generated_messages=[msg], generated_artifacts=[artifact],
                                  diagnostic_values={"option_count": len(options)})


class MockFlightAgent(_SpecialistBase):
    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        if action.action_type == "deliver_flight_options":
            options = self.content_repository.flights_for(task.request.destination)
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.FLIGHT_OPTIONS, options,
                                  request_message_id=action.context["request_message_id"])

        if action.action_type == "flight_clarify_response":
            options = self.content_repository.flights_for(task.request.destination)
            cheapest = min(options, key=lambda f: f["price"])
            answer = {"confirmed_arrival_time": cheapest["arrival_time"],
                      "confirmed_return_departure_time": cheapest["return_departure_time"]}
            part = _data_part(answer, id_factory, created_at)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                                  request_message_id=action.context["request_message_id"])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        raise ValueError(f"MockFlightAgent cannot handle action_type {action.action_type!r}")


class MockHotelAgent(_SpecialistBase):
    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        if action.action_type == "deliver_hotel_options":
            options = self.content_repository.hotels_for(task.request.destination)
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.HOTEL_OPTIONS, options,
                                  request_message_id=action.context["request_message_id"])

        if action.action_type == "hotel_clarify_currency":
            part = _data_part({"requested_pair": f"{task.request.budget_currency}/{task.request.target_currency}"},
                               id_factory, created_at)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        if action.action_type == "deliver_hotel_revision":
            options = self.content_repository.hotels_for(task.request.destination)
            cheapest = min(options, key=lambda h: h["total_price"])
            discounted = dict(cheapest)
            discounted["nightly_price"] = round(cheapest["nightly_price"] * (1 - BUDGET_REVISION_DISCOUNT), 2)
            discounted["total_price"] = round(cheapest["total_price"] * (1 - BUDGET_REVISION_DISCOUNT), 2)
            discounted["description"] = cheapest["description"] + " (revised: loyalty discount applied)"
            prior = action.context["prior_artifact"]
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.HOTEL_OPTIONS, [discounted], version=prior.version + 1,
                                  parent_artifact_ids=[prior.artifact_id],
                                  request_message_id=action.context["request_message_id"])

        raise ValueError(f"MockHotelAgent cannot handle action_type {action.action_type!r}")


class MockCurrencyAgent(_SpecialistBase):
    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        if action.action_type == "deliver_budget_conversion":
            rate = self.content_repository.currency_rate(task.request.budget_currency, task.request.target_currency)
            total_budget_target = task.request.budget_amount * rate
            content = {
                "base_currency": task.request.budget_currency, "target_currency": task.request.target_currency,
                "rate": rate, "total_budget_target_currency": round(total_budget_target, 2),
                "lodging_budget_target_currency": round(total_budget_target * LODGING_BUDGET_FRACTION, 2),
            }
            part = _data_part(content, id_factory, created_at)
            artifact = Artifact(
                artifact_id=id_factory.artifact_id(), task_id=task.task_id, context_id=task.context_id,
                artifact_type=ArtifactType.BUDGET_CONVERSION, producer_id=action.sender_id, version=1,
                part_ids=[part.part_id], record_count=1, created_at=created_at, updated_at=created_at,
            )
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                                  artifact_ids=[artifact.artifact_id], request_message_id=action.context["request_message_id"])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg], generated_artifacts=[artifact],
                                      diagnostic_values=content)

        if action.action_type == "currency_clarify_response":
            rate = self.content_repository.currency_rate(task.request.budget_currency, task.request.target_currency)
            part = _data_part({"confirmed_rate": rate, "pair": f"{task.request.budget_currency}/{task.request.target_currency}"},
                               id_factory, created_at)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id],
                                  request_message_id=action.context["request_message_id"])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        raise ValueError(f"MockCurrencyAgent cannot handle action_type {action.action_type!r}")


class MockToursAgent(_SpecialistBase):
    def handle(self, action, task, artifacts, parts, id_factory, created_at, sequence_index, session_id=None, attack_config=None) -> AgentActionResult:
        if action.action_type == "deliver_tour_options":
            options = self.content_repository.tours_for_in_range(
                task.request.destination, task.request.departure_date, task.request.return_date)
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.TOUR_OPTIONS, options,
                                  request_message_id=action.context["request_message_id"])

        if action.action_type == "tours_clarify_flight":
            conflicting = action.context["conflicting_option_ids"]
            part = _data_part({"conflicting_option_ids": conflicting, "question": "confirm actual arrival/departure times"},
                               id_factory, created_at)
            msg = _reply_message(action, id_factory, created_at, sequence_index, part_ids=[part.part_id])
            return AgentActionResult(generated_parts=[part], generated_messages=[msg])

        if action.action_type == "deliver_tours_schedule_revision":
            prior = action.context["prior_artifact"]
            filtered = action.context["filtered_options"]
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.TOUR_OPTIONS, filtered, version=prior.version + 1,
                                  parent_artifact_ids=[prior.artifact_id])

        if action.action_type == "deliver_tours_integration_revision":
            # Fallback rule: ignore the [departure_date, return_date] filter
            # and return whatever destination tours exist at all -- a
            # deterministic, uniform "widen the search" rule, not a
            # per-fixture special case (only London's fixture data actually
            # has zero in-range matches, triggering this path).
            prior = action.context["prior_artifact"]
            options = self.content_repository.tours_for(task.request.destination)
            return self._deliver(action, task, artifacts, id_factory, created_at, sequence_index,
                                  ArtifactType.TOUR_OPTIONS, options, version=prior.version + 1,
                                  parent_artifact_ids=[prior.artifact_id],
                                  request_message_id=action.context["request_message_id"])

        raise ValueError(f"MockToursAgent cannot handle action_type {action.action_type!r}")


def build_mock_agent_registry(content_repository: ContentRepository) -> dict:
    return {
        "client": MockClient(),
        "travel_coordinator": MockCoordinator(),
        "flight_agent": MockFlightAgent(content_repository),
        "hotel_agent": MockHotelAgent(content_repository),
        "currency_agent": MockCurrencyAgent(content_repository),
        "tours_agent": MockToursAgent(content_repository),
    }
