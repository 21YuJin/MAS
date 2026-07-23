"""
[Step 2-2] Fixed Agent Registry for the travel_a2a_v2 environment.

5 LLM agents, fixed (per the Step 2 instruction -- not variable-node-count):
travel_coordinator, flight_agent, hotel_agent, currency_agent, tours_agent.
Plus one logical, non-LLM node: client (the traveler). MODEL_AGENT_ORDER is
the single source for LLM-agent ordering that a future node-feature tensor /
MLP-AE flatten baseline must use consistently -- client is excluded from it
(it never calls an LLM and has no telemetry to featurize), though it can
still appear in the raw interaction log.
"""
import dataclasses
from typing import List, Optional, Tuple

CLIENT_AGENT_ID = "client"

MODEL_AGENT_ORDER = [
    "travel_coordinator",
    "flight_agent",
    "hotel_agent",
    "currency_agent",
    "tours_agent",
]

ALL_AGENT_IDS = [CLIENT_AGENT_ID] + MODEL_AGENT_ORDER


@dataclasses.dataclass
class AgentCardLite:
    agent_id: str
    display_name: str
    description: str
    skills: List[str] = dataclasses.field(default_factory=list)
    accepted_part_types: List[str] = dataclasses.field(default_factory=list)
    accepted_artifact_types: List[str] = dataclasses.field(default_factory=list)
    produced_artifact_types: List[str] = dataclasses.field(default_factory=list)
    model_name: Optional[str] = None
    system_prompt_version: Optional[str] = None
    config_version: str = "v1"
    llm_enabled: bool = True
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if not self.display_name:
            raise ValueError("display_name must not be empty")

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "description": self.description,
            "skills": list(self.skills),
            "accepted_part_types": list(self.accepted_part_types),
            "accepted_artifact_types": list(self.accepted_artifact_types),
            "produced_artifact_types": list(self.produced_artifact_types),
            "model_name": self.model_name,
            "system_prompt_version": self.system_prompt_version,
            "config_version": self.config_version,
            "llm_enabled": self.llm_enabled,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCardLite":
        return cls(**data)


class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register(self, agent_card: AgentCardLite) -> None:
        if agent_card.agent_id in self._agents:
            raise ValueError(f"agent already registered: {agent_card.agent_id!r}")
        self._agents[agent_card.agent_id] = agent_card

    def get(self, agent_id: str) -> AgentCardLite:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"unknown agent_id: {agent_id!r}") from None

    def contains(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def list_all(self) -> List[AgentCardLite]:
        return list(self._agents.values())

    def list_llm_agents(self) -> List[AgentCardLite]:
        """Ordered by MODEL_AGENT_ORDER (not registration order), so node
        tensor / flatten-baseline ordering is stable regardless of how/when
        agents were registered."""
        order_index = {agent_id: i for i, agent_id in enumerate(MODEL_AGENT_ORDER)}
        llm_agents = [a for a in self._agents.values() if a.llm_enabled]
        return sorted(llm_agents, key=lambda a: order_index.get(a.agent_id, len(MODEL_AGENT_ORDER)))

    def validate_sender_receiver(self, sender_id: str, receiver_id: str) -> Tuple[bool, Optional[str]]:
        """Returns (is_valid, reason) -- reason is None when valid."""
        if not self.contains(sender_id):
            return False, f"unknown sender_id: {sender_id!r}"
        if not self.contains(receiver_id):
            return False, f"unknown receiver_id: {receiver_id!r}"
        if sender_id == receiver_id:
            return False, f"sender_id and receiver_id must differ, both are {sender_id!r}"
        return True, None


def build_default_registry() -> AgentRegistry:
    """
    Standard travel_a2a_v2 registry: client (non-LLM) + the 5 fixed LLM
    agents, per MODEL_AGENT_ORDER. System PROMPT BODIES are not written yet
    (Step 2 scope is object models, not prompt content) -- only
    system_prompt_version is recorded, as a placeholder to version against
    later.
    """
    registry = AgentRegistry()
    registry.register(AgentCardLite(
        agent_id=CLIENT_AGENT_ID, display_name="Client",
        description="The traveler submitting the request. Never calls an LLM.",
        skills=[], accepted_part_types=["text", "data"],
        accepted_artifact_types=["final_travel_plan"], produced_artifact_types=[],
        model_name=None, system_prompt_version=None, llm_enabled=False,
    ))
    registry.register(AgentCardLite(
        agent_id="travel_coordinator", display_name="Travel Coordinator",
        description="Decomposes the travel request, selects specialists, integrates "
                     "their artifacts, resolves conflicts, produces the final itinerary.",
        skills=["task_decomposition", "specialist_selection", "artifact_integration",
                "conflict_resolution", "final_itinerary_generation"],
        accepted_part_types=["text", "data"],
        accepted_artifact_types=["flight_options", "selected_flight", "hotel_options",
                                  "selected_hotel", "exchange_rate", "budget_conversion",
                                  "tour_options", "daily_activity_plan"],
        produced_artifact_types=["integrated_itinerary", "final_travel_plan"],
        model_name="llama3.2", system_prompt_version="travel_coordinator_v1",
    ))
    registry.register(AgentCardLite(
        agent_id="flight_agent", display_name="Flight Agent",
        description="Analyzes flight options, compares schedules and prices.",
        skills=["flight_option_analysis", "schedule_comparison", "price_comparison"],
        accepted_part_types=["text", "data"], accepted_artifact_types=[],
        produced_artifact_types=["flight_options", "selected_flight"],
        model_name="llama3.2", system_prompt_version="flight_agent_v1",
    ))
    registry.register(AgentCardLite(
        agent_id="hotel_agent", display_name="Hotel Agent",
        description="Compares accommodation, matches location, validates lodging budget.",
        skills=["accommodation_comparison", "location_matching", "lodging_budget_validation"],
        accepted_part_types=["text", "data"], accepted_artifact_types=[],
        produced_artifact_types=["hotel_options", "selected_hotel"],
        model_name="llama3.2", system_prompt_version="hotel_agent_v1",
    ))
    registry.register(AgentCardLite(
        agent_id="currency_agent", display_name="Currency Agent",
        description="Processes exchange rates, normalizes currency, converts budget.",
        skills=["exchange_rate_processing", "currency_normalization", "budget_conversion"],
        accepted_part_types=["text", "data"], accepted_artifact_types=[],
        produced_artifact_types=["exchange_rate", "budget_conversion"],
        model_name="llama3.2", system_prompt_version="currency_agent_v1",
    ))
    registry.register(AgentCardLite(
        agent_id="tours_agent", display_name="Tours Agent",
        description="Recommends activities, plans daily schedule, checks schedule conflicts.",
        skills=["activity_recommendation", "daily_schedule_planning", "schedule_conflict_checking"],
        accepted_part_types=["text", "data"], accepted_artifact_types=[],
        produced_artifact_types=["tour_options", "daily_activity_plan"],
        model_name="llama3.2", system_prompt_version="tours_agent_v1",
    ))
    return registry
