"""
[Step 6.5-3] TaskTemplate / TaskInstance schema for the formal_workload/
generation pipeline -- separate from data/travel_a2a/development/'s ad hoc
fixture dicts (fixtures.py), since the formal workload needs machine-checkable
identity (template_id / task_group_id), split membership, and generation
provenance that the 6 development fixtures never needed.

TaskTemplate describes STRUCTURE (constraint types, allowed branches, required
services) -- it is never itself turned into a TravelRequest. TaskInstance is
the frozen, parameter-filled object that Step 6.5B's generator turns into a
TravelRequest/TravelTask pair (mirroring fixtures.py's build_travel_task()).

template_family is a BROADER workload-composition vocabulary than
TravelRequest.task_category (models.py's TaskCategory enum, 6 values): it adds
"hard_normal_trip" as a 7th bucket for Step 6.5-4/6.5-15's "attack-free but
metadata-heavy" normal tasks. A hard_normal_trip instance still carries a real
task_category (one of the existing 6 values) describing what the LLM
actually sees -- template_family/hard_normal_tags are workload-composition
and diagnostic labels, layered ON TOP of the existing TravelRequest schema,
never inside it.

Diagnostic-only fields (Step 6.5-2/6.5-15): expected_normal_branches, split,
hard_normal_tags, generation_seed, generator_version, task_group_id. These
must never reach an LLM prompt or a LightGAE metadata_dict view -- callers
building a TravelRequest from a TaskInstance read only the request-shaped
fields (origin/destination/dates/travelers/budget/preferences/
required_services/task_category/difficulty), exactly as fixtures.py's
build_travel_task() already does for the 6 development fixtures.
"""
import dataclasses
from typing import Any, Dict, List, Optional

from .models import REQUIRED_SERVICE_TYPES, Difficulty, TaskCategory

# [Step 6.5-4] 6 existing TaskCategory values + "hard_normal_trip" -- the 7th
# bucket is a workload-composition label only, never assigned as an
# instance's own TravelRequest.task_category (see module docstring).
TEMPLATE_FAMILIES = tuple(c.value for c in TaskCategory) + ("hard_normal_trip",)

DIFFICULTIES = tuple(d.value for d in Difficulty)

# [Step 6.5-10/6.5-11] primary_group_split membership values.
PRIMARY_SPLITS = ("train", "validation", "test")

# [Step 6.5-15] controlled vocabulary for hard_normal_tags -- diagnostic only.
HARD_NORMAL_TAGS = (
    "long_form_comparison", "multi_step_revision", "high_message_volume",
    "high_token_volume", "repeated_clarification", "complex_schedule",
    "multi_artifact_integration", "multi_currency_conversion",
)


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


@dataclasses.dataclass
class TaskTemplate:
    template_id: str
    template_family: str
    description: str
    required_services: List[str]
    required_preferences: List[str] = dataclasses.field(default_factory=list)
    constraint_types: List[str] = dataclasses.field(default_factory=list)
    allowed_branches: List[str] = dataclasses.field(default_factory=list)
    minimum_difficulty: str = "easy"
    maximum_difficulty: str = "hard"
    generation_rules: Dict[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: str = "travel_a2a_v2_formal_workload_v1"

    def __post_init__(self):
        _require_nonempty(self.template_id, "template_id")
        _require_nonempty(self.description, "description")
        if self.template_family not in TEMPLATE_FAMILIES:
            raise ValueError(f"template_family must be one of {TEMPLATE_FAMILIES}, got {self.template_family!r}")
        if self.minimum_difficulty not in DIFFICULTIES:
            raise ValueError(f"minimum_difficulty must be one of {DIFFICULTIES}, got {self.minimum_difficulty!r}")
        if self.maximum_difficulty not in DIFFICULTIES:
            raise ValueError(f"maximum_difficulty must be one of {DIFFICULTIES}, got {self.maximum_difficulty!r}")
        if DIFFICULTIES.index(self.minimum_difficulty) > DIFFICULTIES.index(self.maximum_difficulty):
            raise ValueError(f"minimum_difficulty {self.minimum_difficulty!r} must not exceed "
                              f"maximum_difficulty {self.maximum_difficulty!r}")
        unknown = set(self.required_services) - REQUIRED_SERVICE_TYPES
        if unknown:
            raise ValueError(f"required_services contains unsupported value(s): {sorted(unknown)}; "
                              f"allowed: {sorted(REQUIRED_SERVICE_TYPES)}")

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id, "template_family": self.template_family,
            "description": self.description, "required_services": list(self.required_services),
            "required_preferences": list(self.required_preferences),
            "constraint_types": list(self.constraint_types), "allowed_branches": list(self.allowed_branches),
            "minimum_difficulty": self.minimum_difficulty, "maximum_difficulty": self.maximum_difficulty,
            "generation_rules": dict(self.generation_rules), "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskTemplate":
        return cls(**data)


@dataclasses.dataclass
class TaskInstance:
    task_instance_id: str
    template_id: str
    task_group_id: str
    origin: str
    destination: str
    departure_date: str
    return_date: str
    travelers: int
    budget_amount: float
    budget_currency: str
    target_currency: str
    required_services: List[str]
    task_category: str
    difficulty: str
    content_bundle_id: str
    flight_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    hotel_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    activity_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # -- diagnostic/provenance only below this line (Step 6.5-2/6.5-15) --
    expected_normal_branches: List[str] = dataclasses.field(default_factory=list)
    split: Optional[str] = None
    hard_normal_tags: List[str] = dataclasses.field(default_factory=list)
    generation_seed: int = 0
    generator_version: str = ""
    schema_version: str = "travel_a2a_v2_formal_workload_v1"

    def __post_init__(self):
        _require_nonempty(self.task_instance_id, "task_instance_id")
        _require_nonempty(self.template_id, "template_id")
        _require_nonempty(self.task_group_id, "task_group_id")
        _require_nonempty(self.origin, "origin")
        _require_nonempty(self.destination, "destination")
        _require_nonempty(self.content_bundle_id, "content_bundle_id")
        if self.travelers < 1:
            raise ValueError(f"travelers must be >= 1, got {self.travelers}")
        if self.budget_amount <= 0:
            raise ValueError(f"budget_amount must be > 0, got {self.budget_amount}")
        self.task_category = TaskCategory(self.task_category).value
        self.difficulty = Difficulty(self.difficulty).value
        unknown = set(self.required_services) - REQUIRED_SERVICE_TYPES
        if unknown:
            raise ValueError(f"required_services contains unsupported value(s): {sorted(unknown)}; "
                              f"allowed: {sorted(REQUIRED_SERVICE_TYPES)}")
        if self.split is not None and self.split not in PRIMARY_SPLITS:
            raise ValueError(f"split must be one of {PRIMARY_SPLITS} or None, got {self.split!r}")
        unknown_tags = set(self.hard_normal_tags) - set(HARD_NORMAL_TAGS)
        if unknown_tags:
            raise ValueError(f"hard_normal_tags contains unsupported value(s): {sorted(unknown_tags)}; "
                              f"allowed: {sorted(HARD_NORMAL_TAGS)}")

    def to_dict(self) -> dict:
        """Full provenance dict -- development/debugging/manifest use only,
        NEVER passed directly as an LLM prompt input or a metadata view (see
        to_travel_request_kwargs())."""
        return {
            "task_instance_id": self.task_instance_id, "template_id": self.template_id,
            "task_group_id": self.task_group_id, "origin": self.origin, "destination": self.destination,
            "departure_date": self.departure_date, "return_date": self.return_date,
            "travelers": self.travelers, "budget_amount": self.budget_amount,
            "budget_currency": self.budget_currency, "target_currency": self.target_currency,
            "required_services": list(self.required_services), "task_category": self.task_category,
            "difficulty": self.difficulty, "content_bundle_id": self.content_bundle_id,
            "flight_preferences": dict(self.flight_preferences), "hotel_preferences": dict(self.hotel_preferences),
            "activity_preferences": dict(self.activity_preferences),
            "expected_normal_branches": list(self.expected_normal_branches), "split": self.split,
            "hard_normal_tags": list(self.hard_normal_tags), "generation_seed": self.generation_seed,
            "generator_version": self.generator_version, "schema_version": self.schema_version,
        }

    def to_travel_request_kwargs(self) -> dict:
        """[Step 6.5-2] Exactly the fields TravelRequest(**kwargs) needs --
        excludes task_instance_id/template_id/task_group_id/content_bundle_id/
        expected_normal_branches/split/hard_normal_tags/generation_seed/
        generator_version/schema_version. This is the ONLY method a formal
        build_travel_task()-equivalent should read from."""
        return {
            "origin": self.origin, "destination": self.destination,
            "departure_date": self.departure_date, "return_date": self.return_date,
            "travelers": self.travelers, "budget_amount": self.budget_amount,
            "budget_currency": self.budget_currency, "target_currency": self.target_currency,
            "flight_preferences": dict(self.flight_preferences), "hotel_preferences": dict(self.hotel_preferences),
            "activity_preferences": dict(self.activity_preferences),
            "required_services": list(self.required_services),
            "task_category": self.task_category, "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskInstance":
        return cls(**data)
