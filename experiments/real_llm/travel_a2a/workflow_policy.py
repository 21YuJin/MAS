"""
[Step 3-5] TravelWorkflowPolicy: decides the NEXT action from current state
(task status, delivered artifacts, message history, event history) -- NOT a
fixed EDGES list traversed in order. `decide()` is a pure function of its
arguments; it holds no state of its own, so the same
(task, artifacts, messages, events, parts) always produces the same decision
regardless of how many times or in what process it's called.

A default sender->receiver sequence DOES exist for a conflict-free task
(client->coordinator->flight->coordinator->hotel->coordinator->currency->
coordinator->tours->coordinator->[integrate]->client) -- that's a property
the rules below happen to produce when every condition check below comes
back negative, not something hardcoded as a step list. Whenever a condition
check is positive (budget exceeded, schedule conflict, missing preference,
empty tour options), a DIFFERENT action is decided instead, changing the
actual trajectory for that task -- see mock_agents.py for what each
action_type does; this module only decides which action_type/sender/receiver
comes next, evaluated fresh from current state on every call.

decide()'s check order (first non-None result wins):
  0. answer the newest unanswered request (LIFO -- a nested clarification
     spawned while an outer request is still open must resolve before the
     outer one can be answered)
  1. submit the task (client -> coordinator), if status == SUBMITTED
  2. once integrating: produce the final artifacts, then complete the task
  3. ask the client about a missing required preference, if not yet asked
  4. start a hotel-budget-revision cycle, if it fires and isn't resolved yet
  5. start a tours/flight schedule-clarification cycle, if it fires and
     isn't resolved yet (or, if already clarified, deliver the resolved v2)
  6. start an integration-revision cycle, if an artifact fails its
     integration contract (e.g. 0 tour options) and isn't resolved yet
  7. delegate to the next required specialist not yet delegated to
  8. otherwise: move to INTEGRATING (every required artifact exists, no
     conflict pending)

Statuses SEARCHING/REVISING/WAITING_FOR_INPUT-after-being-answered are all
handled by the SAME fallthrough (steps 3-8) -- what happens next depends on
which artifacts/conflicts exist NOW, never on which of those labels
task.status currently holds. Only SUBMITTED/INTEGRATING/COMPLETED get their
own branch, because those three are genuinely different questions ("has the
client submitted yet", "is the final plan ready yet", "are we done").
"""
import dataclasses
from typing import List, Optional

from .models import Artifact, ArtifactType, InteractionType, Message, Part, TravelTask
from .status import TaskStatus

SERVICE_PRIORITY = ["flight", "hotel", "currency", "tours"]

SERVICE_TO_DELEGATE_ACTION = {
    "flight": "delegate_flight_search", "hotel": "delegate_hotel_search",
    "currency": "delegate_currency_check", "tours": "delegate_tours_search",
}
SERVICE_TO_AGENT = {
    "flight": "flight_agent", "hotel": "hotel_agent",
    "currency": "currency_agent", "tours": "tours_agent",
}
SERVICE_TO_DELIVER_ACTION = {
    "flight": "deliver_flight_options", "hotel": "deliver_hotel_options",
    "currency": "deliver_budget_conversion", "tours": "deliver_tour_options",
}

# Action types that are pure bookkeeping/internal (produced by the
# coordinator acting on itself) and therefore must NOT go through the normal
# Message/InteractionEvent pipeline -- Message.__post_init__ rejects
# sender_id == receiver_id, and "coordinator generates artifacts" (Step 3-5
# step 10) is given without a "->" arrow in the base-flow spec, unlike every
# other numbered step. mock_runner.py checks this set directly.
INTERNAL_ACTION_TYPES = frozenset({"integrate_itinerary", "move_to_integrating", "resume_planning"})

_REQUEST_TYPES = {InteractionType.TASK_DELEGATION, InteractionType.CLARIFICATION_REQUEST, InteractionType.REVISION_REQUEST}


@dataclasses.dataclass
class WorkflowAction:
    """One decided next step. `context` carries whatever extra state the
    acting agent's handle() needs (mock_agents.py) that doesn't belong on the
    Message/Artifact schema itself (e.g. which artifact_id a revision
    targets); `next_status`, if set, is the TaskStatus the runner should move
    the task to immediately after executing this action."""
    action_type: str
    sender_id: str
    receiver_id: str
    interaction_type: InteractionType
    context: dict = dataclasses.field(default_factory=dict)
    next_status: Optional[TaskStatus] = None


# ══════════════════════════════════════════════════════════════════════════
# State-inspection helpers -- every one is a pure function of the collections
# already passed to decide(); none of them cache or mutate anything.
# ══════════════════════════════════════════════════════════════════════════


def _latest_artifact(artifacts: List[Artifact], artifact_type: ArtifactType) -> Optional[Artifact]:
    matches = [a for a in artifacts if a.artifact_type == artifact_type]
    return max(matches, key=lambda a: a.version) if matches else None


def _artifact_content(artifact: Artifact, parts: List[Part]) -> dict:
    by_id = {p.part_id: p for p in parts}
    if not artifact.part_ids:
        return {}
    return by_id[artifact.part_ids[0]].content


def _messages_matching(messages: List[Message], sender_id=None, receiver_id=None, interaction_type=None) -> List[Message]:
    out = []
    for m in messages:
        if sender_id is not None and m.sender_id != sender_id:
            continue
        if receiver_id is not None and m.receiver_id != receiver_id:
            continue
        if interaction_type is not None and m.interaction_type != interaction_type:
            continue
        out.append(m)
    return out


def _newest_unanswered_request(messages: List[Message]) -> Optional[Message]:
    """LIFO on purpose: a nested clarification (e.g. hotel_agent asking
    currency_agent while coordinator's own revision_request to hotel_agent is
    still open) is created AFTER the outer request and must be resolved
    first -- picking the oldest would try to re-answer the already-open outer
    request while its own dependency is still unresolved."""
    answered_ids = {m.request_message_id for m in messages if m.request_message_id}
    candidates = [m for m in messages if m.interaction_type in _REQUEST_TYPES and m.message_id not in answered_ids]
    return max(candidates, key=lambda m: m.sequence_index) if candidates else None


def _hotel_budget_conflict(task: TravelTask, artifacts: List[Artifact], parts: List[Part]) -> Optional[dict]:
    """None if: no hotel_options yet, no budget_conversion yet, or already
    resolved (a hotel_options v2+ exists) -- otherwise the diagnostic values
    driving the decision. Uniform rule for every destination: lodging budget
    = LODGING_BUDGET_FRACTION (mock_agents.py) of the converted total budget;
    conflict = cheapest delivered hotel option's total_price exceeds it."""
    hotel = _latest_artifact(artifacts, ArtifactType.HOTEL_OPTIONS)
    budget = _latest_artifact(artifacts, ArtifactType.BUDGET_CONVERSION)
    if hotel is None or budget is None or hotel.version > 1:
        return None
    options = _artifact_content(hotel, parts).get("options", [])
    if not options:
        return None
    cheapest_total = min(o["total_price"] for o in options)
    lodging_budget = _artifact_content(budget, parts)["lodging_budget_target_currency"]
    if cheapest_total <= lodging_budget:
        return None
    return {"hotel_artifact": hotel, "cheapest_total": cheapest_total, "lodging_budget": lodging_budget}


def _tour_schedule_conflict(task: TravelTask, artifacts: List[Artifact], parts: List[Part]) -> Optional[dict]:
    """None if: no tour_options yet (or empty -- that's the integration-
    revision branch's concern, not this one's), no flight_options yet,
    already resolved (v2+ exists), or no option actually conflicts."""
    tours = _latest_artifact(artifacts, ArtifactType.TOUR_OPTIONS)
    flight = _latest_artifact(artifacts, ArtifactType.FLIGHT_OPTIONS)
    if tours is None or flight is None or tours.version > 1:
        return None
    tour_options = _artifact_content(tours, parts).get("options", [])
    flight_options = _artifact_content(flight, parts).get("options", [])
    if not tour_options or not flight_options:
        return None
    cheapest_flight = min(flight_options, key=lambda f: f["price"])
    arrival_date = cheapest_flight["arrival_time"][:10]
    departure_date = cheapest_flight["return_departure_time"][:10]
    conflicting = []
    for opt in tour_options:
        if opt["date"] == arrival_date and opt["start_time"] < cheapest_flight["arrival_time"]:
            conflicting.append(opt["option_id"])
        elif opt["date"] == departure_date and opt["end_time"] > cheapest_flight["return_departure_time"]:
            conflicting.append(opt["option_id"])
    if not conflicting:
        return None
    return {"tours_artifact": tours, "conflicting_option_ids": conflicting}


def _integration_contract_failure(artifacts: List[Artifact]) -> Optional[dict]:
    """Only tour_options' "0 options delivered" case is implemented (per Step
    3-6.D's example list) -- hotel-stay-date-mismatch / no-tour-after-arrival
    are left for a future round, since none of the 6 fixtures need them to
    exercise this branch."""
    tours = _latest_artifact(artifacts, ArtifactType.TOUR_OPTIONS)
    if tours is None or tours.record_count > 0 or tours.version > 1:
        return None
    return {"artifact": tours, "reason": "zero_tour_options_in_range"}


class TravelWorkflowPolicy:
    def decide(self, task: TravelTask, artifacts: List[Artifact], messages: List[Message],
               events: List[dict], parts: Optional[List[Part]] = None) -> Optional[WorkflowAction]:
        """Returns the next WorkflowAction, or None once the task is
        COMPLETED. `parts` is not in the illustrative signature from the Step
        3 instruction but is required in practice: budget/schedule conflict
        detection needs the actual content values (prices/times), not just
        artifact metadata (record_count/version) -- see the module docstring.
        mock_runner.py additionally enforces a step-count safety cap in case
        a future rule change ever creates a non-terminating loop."""
        parts = parts or []
        if task.status == TaskStatus.COMPLETED:
            return None

        unanswered = _newest_unanswered_request(messages)
        if unanswered is not None:
            return self._answer(unanswered, task, artifacts, messages, parts)

        if task.status == TaskStatus.SUBMITTED:
            return WorkflowAction(action_type="submit_task", sender_id="client", receiver_id="travel_coordinator",
                                   interaction_type=InteractionType.TASK_SUBMISSION,
                                   context={"task_id": task.task_id, "context_id": task.context_id},
                                   next_status=TaskStatus.PLANNING)

        if task.status == TaskStatus.WAITING_FOR_INPUT:
            # No unanswered request reached this point (step 0 above already
            # handles "still waiting") -- the client has answered. status.py's
            # transition table only allows WAITING_FOR_INPUT -> PLANNING
            # (never straight to SEARCHING), so this explicit hop is required
            # before the unified checks below can delegate again.
            return WorkflowAction(action_type="resume_planning", sender_id="travel_coordinator",
                                   receiver_id="travel_coordinator", interaction_type=InteractionType.STATUS_UPDATE,
                                   context={"task_id": task.task_id, "context_id": task.context_id},
                                   next_status=TaskStatus.PLANNING)

        if task.status == TaskStatus.INTEGRATING:
            final_plan = _latest_artifact(artifacts, ArtifactType.FINAL_TRAVEL_PLAN)
            if final_plan is None:
                return WorkflowAction(action_type="integrate_itinerary", sender_id="travel_coordinator",
                                       receiver_id="travel_coordinator", interaction_type=InteractionType.STATUS_UPDATE,
                                       context={"task_id": task.task_id, "context_id": task.context_id})
            return WorkflowAction(action_type="task_completion", sender_id="travel_coordinator", receiver_id="client",
                                   interaction_type=InteractionType.TASK_COMPLETION,
                                   context={"task_id": task.task_id, "context_id": task.context_id},
                                   next_status=TaskStatus.COMPLETED)

        # PLANNING / SEARCHING / REVISING / (WAITING_FOR_INPUT already
        # answered, since step 0 above handles "still waiting") all share the
        # checks below -- see module docstring.

        if not task.request.hotel_preferences and not _messages_matching(
                messages, sender_id="travel_coordinator", receiver_id="client",
                interaction_type=InteractionType.CLARIFICATION_REQUEST):
            return WorkflowAction(action_type="request_client_clarification", sender_id="travel_coordinator",
                                   receiver_id="client", interaction_type=InteractionType.CLARIFICATION_REQUEST,
                                   context={"task_id": task.task_id, "context_id": task.context_id},
                                   next_status=TaskStatus.WAITING_FOR_INPUT)

        budget_conflict = _hotel_budget_conflict(task, artifacts, parts)
        if budget_conflict is not None:
            hotel = budget_conflict["hotel_artifact"]
            return WorkflowAction(
                action_type="request_hotel_revision", sender_id="travel_coordinator", receiver_id="hotel_agent",
                interaction_type=InteractionType.REVISION_REQUEST,
                context={"task_id": task.task_id, "context_id": task.context_id, "target_artifact_id": hotel.artifact_id,
                         "reason_payload": {"reason": "budget_exceeded", "cheapest_total": budget_conflict["cheapest_total"],
                                             "lodging_budget": budget_conflict["lodging_budget"]}},
                next_status=TaskStatus.REVISING)

        schedule_conflict = _tour_schedule_conflict(task, artifacts, parts)
        if schedule_conflict is not None:
            already_asked = _messages_matching(messages, sender_id="tours_agent", receiver_id="flight_agent",
                                                interaction_type=InteractionType.CLARIFICATION_REQUEST)
            if not already_asked:
                return WorkflowAction(
                    action_type="tours_clarify_flight", sender_id="tours_agent", receiver_id="flight_agent",
                    interaction_type=InteractionType.CLARIFICATION_REQUEST,
                    context={"task_id": task.task_id, "context_id": task.context_id,
                             "conflicting_option_ids": schedule_conflict["conflicting_option_ids"]},
                    next_status=TaskStatus.REVISING)
            prior = schedule_conflict["tours_artifact"]
            all_options = _artifact_content(prior, parts)["options"]
            filtered = [o for o in all_options if o["option_id"] not in schedule_conflict["conflicting_option_ids"]]
            return WorkflowAction(
                action_type="deliver_tours_schedule_revision", sender_id="tours_agent", receiver_id="travel_coordinator",
                interaction_type=InteractionType.ARTIFACT_DELIVERY,
                context={"task_id": task.task_id, "context_id": task.context_id, "prior_artifact": prior,
                         "filtered_options": filtered})

        integration_failure = _integration_contract_failure(artifacts)
        if integration_failure is not None:
            already_requested = _messages_matching(messages, sender_id="travel_coordinator", receiver_id="tours_agent",
                                                     interaction_type=InteractionType.REVISION_REQUEST)
            if not already_requested:
                return WorkflowAction(
                    action_type="request_integration_revision", sender_id="travel_coordinator", receiver_id="tours_agent",
                    interaction_type=InteractionType.REVISION_REQUEST,
                    context={"task_id": task.task_id, "context_id": task.context_id,
                             "target_artifact_id": integration_failure["artifact"].artifact_id,
                             "reason_payload": {"reason": integration_failure["reason"]}},
                    next_status=TaskStatus.REVISING)
            # already requested but still failing -- shouldn't happen with the
            # mock fallback rule (always widens the search); fall through
            # rather than loop forever.

        next_service = self._next_service_to_delegate(task, artifacts, messages)
        if next_service is not None:
            return WorkflowAction(action_type=SERVICE_TO_DELEGATE_ACTION[next_service], sender_id="travel_coordinator",
                                   receiver_id=SERVICE_TO_AGENT[next_service], interaction_type=InteractionType.TASK_DELEGATION,
                                   context={"task_id": task.task_id, "context_id": task.context_id},
                                   next_status=TaskStatus.SEARCHING)

        return WorkflowAction(action_type="move_to_integrating", sender_id="travel_coordinator",
                               receiver_id="travel_coordinator", interaction_type=InteractionType.STATUS_UPDATE,
                               context={"task_id": task.task_id, "context_id": task.context_id},
                               next_status=TaskStatus.INTEGRATING)

    def _next_service_to_delegate(self, task, artifacts, messages) -> Optional[str]:
        for service in SERVICE_PRIORITY:
            if service not in task.request.required_services:
                continue
            already_delegated = _messages_matching(messages, sender_id="travel_coordinator",
                                                     receiver_id=SERVICE_TO_AGENT[service],
                                                     interaction_type=InteractionType.TASK_DELEGATION)
            if not already_delegated:
                return service
        return None

    def _answer(self, request: Message, task, artifacts, messages, parts) -> WorkflowAction:
        ctx = {"task_id": task.task_id, "context_id": task.context_id, "request_message_id": request.message_id}

        if request.interaction_type == InteractionType.TASK_DELEGATION:
            service = next(s for s, agent in SERVICE_TO_AGENT.items() if agent == request.receiver_id)
            return WorkflowAction(action_type=SERVICE_TO_DELIVER_ACTION[service], sender_id=request.receiver_id,
                                   receiver_id=request.sender_id, interaction_type=InteractionType.ARTIFACT_DELIVERY,
                                   context=ctx)

        if request.interaction_type == InteractionType.CLARIFICATION_REQUEST:
            if request.sender_id == "travel_coordinator" and request.receiver_id == "client":
                return WorkflowAction(action_type="client_clarification_response", sender_id="client",
                                       receiver_id="travel_coordinator", interaction_type=InteractionType.CLARIFICATION_RESPONSE,
                                       context=ctx)
            if request.sender_id == "hotel_agent" and request.receiver_id == "currency_agent":
                return WorkflowAction(action_type="currency_clarify_response", sender_id="currency_agent",
                                       receiver_id="hotel_agent", interaction_type=InteractionType.CLARIFICATION_RESPONSE,
                                       context=ctx)
            if request.sender_id == "tours_agent" and request.receiver_id == "flight_agent":
                return WorkflowAction(action_type="flight_clarify_response", sender_id="flight_agent",
                                       receiver_id="tours_agent", interaction_type=InteractionType.CLARIFICATION_RESPONSE,
                                       context=ctx)
            raise AssertionError(f"no clarification_response rule for {request.sender_id!r} -> {request.receiver_id!r}")

        if request.interaction_type == InteractionType.REVISION_REQUEST:
            if request.receiver_id == "hotel_agent":
                # Step 3-6.A: hotel always re-confirms the rate with currency
                # before delivering a revision, even in this ordering where
                # it already received one earlier (treated as a deliberate
                # double-check, not redundant per the Step 3 instruction).
                already_clarified = _messages_matching(messages, sender_id="hotel_agent", receiver_id="currency_agent",
                                                         interaction_type=InteractionType.CLARIFICATION_REQUEST)
                if not already_clarified:
                    return WorkflowAction(action_type="hotel_clarify_currency", sender_id="hotel_agent",
                                           receiver_id="currency_agent", interaction_type=InteractionType.CLARIFICATION_REQUEST,
                                           context={"task_id": task.task_id, "context_id": task.context_id})
                prior = _latest_artifact(artifacts, ArtifactType.HOTEL_OPTIONS)
                return WorkflowAction(action_type="deliver_hotel_revision", sender_id="hotel_agent",
                                       receiver_id="travel_coordinator", interaction_type=InteractionType.ARTIFACT_DELIVERY,
                                       context=dict(ctx, prior_artifact=prior))
            if request.receiver_id == "tours_agent":
                prior = _latest_artifact(artifacts, ArtifactType.TOUR_OPTIONS)
                return WorkflowAction(action_type="deliver_tours_integration_revision", sender_id="tours_agent",
                                       receiver_id="travel_coordinator", interaction_type=InteractionType.ARTIFACT_DELIVERY,
                                       context=dict(ctx, prior_artifact=prior))
            raise AssertionError(f"no revision-delivery rule for receiver {request.receiver_id!r}")

        raise AssertionError(f"unhandled request interaction_type: {request.interaction_type!r}")
