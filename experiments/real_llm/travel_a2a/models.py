"""
[Step 2-4] Core object models for the travel_a2a_v2 environment: TravelRequest,
TravelTask, Part, Message, Artifact, InteractionEvent.

Common principles applied to every class here (per the Step 2 instruction):
  - JSON serializable via explicit to_dict()/from_dict() (not bare
    dataclasses.asdict(), so enum encoding and nested-object handling are
    unambiguous rather than relying on json.dumps' str-subclass behavior).
  - schema_version on every object.
  - IDs must not be empty strings (checked in __post_init__).
  - Timestamps are ISO 8601 strings (matches this codebase's existing
    convention -- see runtime/ollama_client.py's start_timestamp/end_timestamp).
  - List fields use default_factory, never a mutable default.
  - content vs. metadata are kept structurally separate: see
    to_metadata_dict() on Part/Message/Artifact/InteractionEvent and
    FORBIDDEN_METADATA_KEYS below.

No Ollama calls, no workflow execution, no attack payloads here -- this file
only defines the object shapes a future execution engine will produce.
"""
import dataclasses
import datetime as dt
import enum
import json
import re
from typing import Any, Dict, List, Optional

from .status import TaskStatus

# ══════════════════════════════════════════════════════════════════════════
# Enums / controlled vocabularies
# ══════════════════════════════════════════════════════════════════════════


class TaskCategory(str, enum.Enum):
    BASIC_TRIP = "basic_trip"
    BUDGET_TRIP = "budget_trip"
    FAMILY_TRIP = "family_trip"
    BUSINESS_TRIP = "business_trip"
    ACTIVITY_FOCUSED_TRIP = "activity_focused_trip"
    MULTI_CONSTRAINT_TRIP = "multi_constraint_trip"


class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PartType(str, enum.Enum):
    TEXT = "text"
    DATA = "data"
    FILE = "file"
    REFERENCE = "reference"


class SourceType(str, enum.Enum):
    USER_REQUEST = "user_request"
    EXTERNAL_CONTENT = "external_content"
    AGENT_GENERATED = "agent_generated"
    ARTIFACT_REFERENCE = "artifact_reference"
    SYSTEM_METADATA = "system_metadata"


class InteractionType(str, enum.Enum):
    TASK_SUBMISSION = "task_submission"
    TASK_DELEGATION = "task_delegation"
    ARTIFACT_DELIVERY = "artifact_delivery"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    REVISION_REQUEST = "revision_request"
    STATUS_UPDATE = "status_update"
    TASK_COMPLETION = "task_completion"


class ArtifactType(str, enum.Enum):
    FLIGHT_OPTIONS = "flight_options"
    SELECTED_FLIGHT = "selected_flight"
    HOTEL_OPTIONS = "hotel_options"
    SELECTED_HOTEL = "selected_hotel"
    EXCHANGE_RATE = "exchange_rate"
    BUDGET_CONVERSION = "budget_conversion"
    TOUR_OPTIONS = "tour_options"
    DAILY_ACTIVITY_PLAN = "daily_activity_plan"
    INTEGRATED_ITINERARY = "integrated_itinerary"
    FINAL_TRAVEL_PLAN = "final_travel_plan"


# required_services vocabulary -- tied directly to the 4 non-coordinator LLM
# specialist agents' domains (flight_agent/hotel_agent/currency_agent/tours_agent).
REQUIRED_SERVICE_TYPES = {"flight", "hotel", "currency", "tours"}

_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")

# [Step 2-7] Fields that must NEVER appear in a *_to_metadata_dict() output --
# these are ground-truth/diagnostic values, and leaking any of them into the
# LightGAE feature-extraction path would be label leakage, not a feature.
# Tests assert this set has empty intersection with every to_metadata_dict()
# key set (see tests/test_travel_a2a.py, check 15).
FORBIDDEN_METADATA_KEYS = {
    "content", "injection_present", "attack_id", "attack_goal",
    "goal_success", "indicator_observed", "propagation_observed",
    "workflow_changed", "condition", "ground_truth_label",
    # [Step 5-13] attack-scenario diagnostic/ground-truth fields -- same
    # leakage risk as the original set above, extended once AttackConfig/
    # AttackExecutionDiagnostics (attack_models.py) existed to define them.
    "attack_family", "affected_agent_ids", "affected_artifact_ids",
    "expected_propagation_path", "evaluator_result", "evaluator_id",
    "evaluator_confidence", "evaluator_evidence", "indicator_patterns",
}


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _validate_date_range(departure_date: str, return_date: str) -> None:
    dep = dt.date.fromisoformat(departure_date)
    ret = dt.date.fromisoformat(return_date)
    if ret < dep:
        raise ValueError(f"return_date ({return_date}) must be >= departure_date ({departure_date})")


def _validate_currency_code(code: str, field_name: str) -> None:
    if not code or not _CURRENCY_CODE_RE.match(code):
        raise ValueError(f"{field_name} must be a 3-letter uppercase ISO 4217 currency code, got {code!r}")


def _estimate_size_bytes(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return len(json.dumps(content, ensure_ascii=False).encode("utf-8"))


# ══════════════════════════════════════════════════════════════════════════
# A. TravelRequest
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class TravelRequest:
    origin: str
    destination: str
    departure_date: str   # ISO 8601 date, "YYYY-MM-DD"
    return_date: str
    travelers: int
    budget_amount: float
    budget_currency: str
    target_currency: str
    flight_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    hotel_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    activity_preferences: Dict[str, Any] = dataclasses.field(default_factory=dict)
    required_services: List[str] = dataclasses.field(default_factory=list)
    task_category: TaskCategory = TaskCategory.BASIC_TRIP
    difficulty: Difficulty = Difficulty.EASY
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        _require_nonempty(self.origin, "origin")
        _require_nonempty(self.destination, "destination")
        _validate_date_range(self.departure_date, self.return_date)
        if self.travelers < 1:
            raise ValueError(f"travelers must be >= 1, got {self.travelers}")
        if self.budget_amount <= 0:
            raise ValueError(f"budget_amount must be > 0, got {self.budget_amount}")
        _validate_currency_code(self.budget_currency, "budget_currency")
        _validate_currency_code(self.target_currency, "target_currency")
        unknown = set(self.required_services) - REQUIRED_SERVICE_TYPES
        if unknown:
            raise ValueError(
                f"required_services contains unsupported value(s): {sorted(unknown)}; "
                f"allowed: {sorted(REQUIRED_SERVICE_TYPES)}")
        self.task_category = TaskCategory(self.task_category)
        self.difficulty = Difficulty(self.difficulty)

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.departure_date,
            "return_date": self.return_date,
            "travelers": self.travelers,
            "budget_amount": self.budget_amount,
            "budget_currency": self.budget_currency,
            "target_currency": self.target_currency,
            "flight_preferences": dict(self.flight_preferences),
            "hotel_preferences": dict(self.hotel_preferences),
            "activity_preferences": dict(self.activity_preferences),
            "required_services": list(self.required_services),
            "task_category": self.task_category.value,
            "difficulty": self.difficulty.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TravelRequest":
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════
# B. TravelTask
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class TravelTask:
    task_id: str
    context_id: str
    request: TravelRequest
    status: TaskStatus = TaskStatus.SUBMITTED
    parent_task_id: Optional[str] = None
    task_type: str = "travel_planning"
    required_artifact_types: List[str] = dataclasses.field(default_factory=list)
    produced_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    assigned_agent_ids: List[str] = dataclasses.field(default_factory=list)
    condition: str = "normal"   # "normal" | "attack"
    injection_present: bool = False
    attack_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        _require_nonempty(self.task_id, "task_id")
        _require_nonempty(self.context_id, "context_id")
        if isinstance(self.request, dict):
            self.request = TravelRequest.from_dict(self.request)
        self.status = TaskStatus(self.status)
        if self.condition not in ("normal", "attack"):
            raise ValueError(f"condition must be 'normal' or 'attack', got {self.condition!r}")
        # [analysis_plan.md §3 principle carried into this schema] condition
        # and injection_present/attack_id must agree with each other -- a
        # session's ground truth is fixed by pool membership, never allowed
        # to silently drift out of sync across these three fields.
        if self.condition == "normal":
            if self.injection_present:
                raise ValueError("injection_present must be False when condition == 'normal'")
            if self.attack_id is not None:
                raise ValueError("attack_id must be None when condition == 'normal'")
        else:
            if not self.injection_present:
                raise ValueError("injection_present must be True when condition == 'attack'")
            if self.attack_id is None:
                raise ValueError("attack_id must not be None when condition == 'attack'")

    @property
    def ground_truth_label(self) -> int:
        return int(self.injection_present)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "context_id": self.context_id,
            "request": self.request.to_dict(),
            "status": self.status.value,
            "parent_task_id": self.parent_task_id,
            "task_type": self.task_type,
            "required_artifact_types": list(self.required_artifact_types),
            "produced_artifact_ids": list(self.produced_artifact_ids),
            "assigned_agent_ids": list(self.assigned_agent_ids),
            "condition": self.condition,
            "injection_present": self.injection_present,
            "attack_id": self.attack_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": dict(self.provenance),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TravelTask":
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════
# C. Part
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Part:
    part_id: str
    part_type: PartType
    mime_type: str
    content: Any
    source_type: SourceType
    source_id: Optional[str] = None
    provenance_id: Optional[str] = None
    injection_present: bool = False
    attack_id: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: str = ""
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        _require_nonempty(self.part_id, "part_id")
        self.part_type = PartType(self.part_type)
        self.source_type = SourceType(self.source_type)
        if self.size_bytes is None:
            self.size_bytes = _estimate_size_bytes(self.content)
        if not self.injection_present and self.attack_id is not None:
            raise ValueError("attack_id must be None when injection_present is False")

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "part_type": self.part_type.value,
            "mime_type": self.mime_type,
            "content": self.content,
            "size_bytes": self.size_bytes,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "provenance_id": self.provenance_id,
            "injection_present": self.injection_present,
            "attack_id": self.attack_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Part":
        return cls(**data)

    def to_metadata_dict(self) -> dict:
        """
        [Step 2-7] Content-free view for LightGAE feature extraction. Must
        never include content/injection_present/attack_id or any other
        FORBIDDEN_METADATA_KEYS entry -- those may only ever live on the raw
        object (self), never on this view.
        """
        return {
            "part_id": self.part_id,
            "part_type": self.part_type.value,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "source_type": self.source_type.value,
            "schema_version": self.schema_version,
        }


# ══════════════════════════════════════════════════════════════════════════
# D. Message
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Message:
    message_id: str
    task_id: str
    context_id: str
    sender_id: str
    receiver_id: str
    interaction_type: InteractionType
    role: str
    part_ids: List[str] = dataclasses.field(default_factory=list)
    artifact_ids: List[str] = dataclasses.field(default_factory=list)
    request_message_id: Optional[str] = None
    sequence_index: int = 0
    created_at: str = ""
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        for name in ("message_id", "task_id", "context_id", "sender_id", "receiver_id"):
            _require_nonempty(getattr(self, name), name)
        # Structural default (no registry access needed): reject a
        # self-addressed message outright. Whether sender/receiver are
        # actually KNOWN agents is a registry-dependent check -- see
        # validation.validate_message().
        if self.sender_id == self.receiver_id:
            raise ValueError(f"sender_id and receiver_id must differ, both are {self.sender_id!r}")
        self.interaction_type = InteractionType(self.interaction_type)
        if self.sequence_index < 0:
            raise ValueError(f"sequence_index must be >= 0, got {self.sequence_index}")

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "interaction_type": self.interaction_type.value,
            "role": self.role,
            "part_ids": list(self.part_ids),
            "artifact_ids": list(self.artifact_ids),
            "request_message_id": self.request_message_id,
            "sequence_index": self.sequence_index,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)

    def to_metadata_dict(self) -> dict:
        """[Step 2-7] Content-free view -- Message never carries `content`
        itself (that lives on its referenced Parts), but is still routed
        through an explicit view for consistency with Part/Artifact/
        InteractionEvent and so a future field addition can't silently leak."""
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "interaction_type": self.interaction_type.value,
            "part_ids": list(self.part_ids),
            "artifact_ids": list(self.artifact_ids),
            "sequence_index": self.sequence_index,
            "schema_version": self.schema_version,
        }


# ══════════════════════════════════════════════════════════════════════════
# E. Artifact
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Artifact:
    artifact_id: str
    task_id: str
    context_id: str
    artifact_type: ArtifactType
    producer_id: str
    version: int = 1
    parent_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    source_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    part_ids: List[str] = dataclasses.field(default_factory=list)
    size_bytes: Optional[int] = None
    record_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        for name in ("artifact_id", "task_id", "context_id", "producer_id"):
            _require_nonempty(getattr(self, name), name)
        self.artifact_type = ArtifactType(self.artifact_type)
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")
        if self.record_count < 0:
            raise ValueError(f"record_count must be >= 0, got {self.record_count}")
        if self.artifact_id in self.parent_artifact_ids:
            raise ValueError(f"artifact {self.artifact_id!r} cannot list itself as its own parent_artifact_id")
        if self.artifact_id in self.source_artifact_ids:
            raise ValueError(f"artifact {self.artifact_id!r} cannot list itself as its own source_artifact_id")

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "artifact_type": self.artifact_type.value,
            "producer_id": self.producer_id,
            "version": self.version,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "part_ids": list(self.part_ids),
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Artifact":
        return cls(**data)

    def to_metadata_dict(self) -> dict:
        """[Step 2-7] Content-free view -- Artifact itself never carries raw
        `content` (that lives on its constituent Parts); exists for the same
        consistency/future-proofing reason as Message.to_metadata_dict()."""
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "artifact_type": self.artifact_type.value,
            "producer_id": self.producer_id,
            "version": self.version,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "part_ids": list(self.part_ids),
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "schema_version": self.schema_version,
        }


# ══════════════════════════════════════════════════════════════════════════
# F. InteractionEvent -- one directed edge in the future raw temporal
#    directed multigraph (not built in this Step).
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class InteractionEvent:
    event_id: str
    event_index: int
    session_id: str
    task_id: str
    context_id: str
    sender_id: str
    receiver_id: str
    interaction_type: InteractionType
    message_id: Optional[str] = None
    part_ids: List[str] = dataclasses.field(default_factory=list)
    artifact_ids: List[str] = dataclasses.field(default_factory=list)
    status_before: Optional[TaskStatus] = None
    status_after: Optional[TaskStatus] = None
    status_transition_valid: Optional[bool] = None
    start_timestamp: str = ""
    end_timestamp: str = ""
    previous_event_timestamp: Optional[str] = None
    llm_called: bool = False
    model_name: Optional[str] = None
    retry_count: int = 0
    error_flag: bool = False
    done_reason: Optional[str] = None
    raw_ollama_telemetry: Optional[dict] = None
    # [Step 3-7] Provenance for WHERE start/end/previous timestamps came from --
    # "deterministic_mock" (fixed-increment DeterministicClock, mock_runner.py)
    # vs. "ollama_runtime" (actual wall-clock measurements, runtime/ollama_client.py).
    # Required so mock timing can never be silently mixed into real
    # normal/attack telemetry -- a caller can filter/assert on this field
    # instead of having to infer origin from timestamp shape.
    timing_source: Optional[str] = None
    schema_version: str = "travel_a2a_v1"

    _VALID_TIMING_SOURCES = ("deterministic_mock", "ollama_runtime")

    def __post_init__(self):
        for name in ("event_id", "session_id", "task_id", "context_id", "sender_id", "receiver_id"):
            _require_nonempty(getattr(self, name), name)
        if self.event_index < 0:
            raise ValueError(f"event_index must be >= 0, got {self.event_index}")
        self.interaction_type = InteractionType(self.interaction_type)
        if self.status_before is not None:
            self.status_before = TaskStatus(self.status_before)
        if self.status_after is not None:
            self.status_after = TaskStatus(self.status_after)
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be >= 0, got {self.retry_count}")
        if self.timing_source is not None and self.timing_source not in self._VALID_TIMING_SOURCES:
            raise ValueError(f"timing_source must be one of {self._VALID_TIMING_SOURCES} or None, "
                              f"got {self.timing_source!r}")

    @property
    def wall_clock_latency_ms(self) -> Optional[float]:
        """Computed from start_timestamp/end_timestamp -- not stored
        redundantly. A raw collector's own measured latency (if any) belongs
        inside raw_ollama_telemetry, untouched."""
        if not self.start_timestamp or not self.end_timestamp:
            return None
        start = dt.datetime.fromisoformat(self.start_timestamp)
        end = dt.datetime.fromisoformat(self.end_timestamp)
        return (end - start).total_seconds() * 1000.0

    @property
    def time_since_previous_event_ms(self) -> Optional[float]:
        if not self.previous_event_timestamp or not self.start_timestamp:
            return None
        prev = dt.datetime.fromisoformat(self.previous_event_timestamp)
        start = dt.datetime.fromisoformat(self.start_timestamp)
        return (start - prev).total_seconds() * 1000.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "interaction_type": self.interaction_type.value,
            "message_id": self.message_id,
            "part_ids": list(self.part_ids),
            "artifact_ids": list(self.artifact_ids),
            "status_before": (self.status_before.value if self.status_before is not None else None),
            "status_after": (self.status_after.value if self.status_after is not None else None),
            "status_transition_valid": self.status_transition_valid,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "previous_event_timestamp": self.previous_event_timestamp,
            "llm_called": self.llm_called,
            "model_name": self.model_name,
            "retry_count": self.retry_count,
            "error_flag": self.error_flag,
            "done_reason": self.done_reason,
            "raw_ollama_telemetry": self.raw_ollama_telemetry,
            "timing_source": self.timing_source,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InteractionEvent":
        return cls(**data)

    def to_metadata_dict(self) -> dict:
        """
        [Step 2-7] Content-free view. Deliberately excludes raw_ollama_telemetry
        (may embed raw response text via its own 'text'/'raw_response' fields,
        see runtime/ollama_client.py) so the feature-extraction path never has
        to know how to redact that structure itself -- it just never receives
        it. message_id/part_ids/artifact_ids are foreign keys, not content, so
        they stay.
        """
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "context_id": self.context_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "interaction_type": self.interaction_type.value,
            "part_ids": list(self.part_ids),
            "artifact_ids": list(self.artifact_ids),
            "status_before": (self.status_before.value if self.status_before is not None else None),
            "status_after": (self.status_after.value if self.status_after is not None else None),
            "status_transition_valid": self.status_transition_valid,
            "llm_called": self.llm_called,
            "model_name": self.model_name,
            "retry_count": self.retry_count,
            "error_flag": self.error_flag,
            "done_reason": self.done_reason,
            "wall_clock_latency_ms": self.wall_clock_latency_ms,
            "time_since_previous_event_ms": self.time_since_previous_event_ms,
            "timing_source": self.timing_source,
            "schema_version": self.schema_version,
        }


# ══════════════════════════════════════════════════════════════════════════
# G. AgentCallRecord -- [Step 4-1] one Agent's own internal execution (LLM
# call or otherwise), as opposed to InteractionEvent (one inter-agent
# COMMUNICATION -- a graph edge). An agent handling one WorkflowAction
# produces exactly one AgentCallRecord always, plus a Message/InteractionEvent
# pair only when the action actually communicates with another agent (not for
# purely-internal work like artifact_integration -- see workflow_policy.py's
# INTERNAL_ACTION_TYPES). Node-level telemetry (prompt_eval_count/eval_count/
# durations) belongs here, never on InteractionEvent, which only ever
# describes the communication itself.
# ══════════════════════════════════════════════════════════════════════════

_VALID_TIMING_SOURCES_CALL = ("deterministic_mock", "ollama_runtime")


@dataclasses.dataclass
class AgentCallRecord:
    call_id: str
    session_id: str
    task_id: str
    context_id: str
    agent_id: str
    action_type: str
    triggering_message_id: Optional[str] = None
    input_part_ids: List[str] = dataclasses.field(default_factory=list)
    input_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    output_part_ids: List[str] = dataclasses.field(default_factory=list)
    output_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    call_start_timestamp: str = ""
    call_end_timestamp: str = ""
    wall_clock_latency_ms: Optional[float] = None
    llm_called: bool = False
    model_name: Optional[str] = None
    model_digest: Optional[str] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    done_reason: Optional[str] = None
    retry_count: int = 0
    error_flag: bool = False
    error_type: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    prompt_config_version: Optional[str] = None
    agent_config_version: Optional[str] = None
    raw_ollama_telemetry: Optional[dict] = None
    timing_source: Optional[str] = None
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        for name in ("call_id", "session_id", "task_id", "context_id", "agent_id", "action_type"):
            _require_nonempty(getattr(self, name), name)
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be >= 0, got {self.retry_count}")
        if self.timing_source is not None and self.timing_source not in _VALID_TIMING_SOURCES_CALL:
            raise ValueError(f"timing_source must be one of {_VALID_TIMING_SOURCES_CALL} or None, "
                              f"got {self.timing_source!r}")

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id, "session_id": self.session_id, "task_id": self.task_id,
            "context_id": self.context_id, "agent_id": self.agent_id, "action_type": self.action_type,
            "triggering_message_id": self.triggering_message_id,
            "input_part_ids": list(self.input_part_ids), "input_artifact_ids": list(self.input_artifact_ids),
            "output_part_ids": list(self.output_part_ids), "output_artifact_ids": list(self.output_artifact_ids),
            "call_start_timestamp": self.call_start_timestamp, "call_end_timestamp": self.call_end_timestamp,
            "wall_clock_latency_ms": self.wall_clock_latency_ms,
            "llm_called": self.llm_called, "model_name": self.model_name, "model_digest": self.model_digest,
            "prompt_eval_count": self.prompt_eval_count, "eval_count": self.eval_count,
            "prompt_eval_duration": self.prompt_eval_duration, "eval_duration": self.eval_duration,
            "total_duration": self.total_duration, "load_duration": self.load_duration,
            "done_reason": self.done_reason, "retry_count": self.retry_count,
            "error_flag": self.error_flag, "error_type": self.error_type,
            "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed,
            "prompt_config_version": self.prompt_config_version, "agent_config_version": self.agent_config_version,
            "raw_ollama_telemetry": self.raw_ollama_telemetry, "timing_source": self.timing_source,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCallRecord":
        return cls(**data)

    def to_metadata_dict(self) -> dict:
        """
        [Step 4-1] Content-free view. Excludes raw_ollama_telemetry entirely
        (it embeds prompt/response text via ask_ollama's own 'text'/
        'raw_response' fields, same reasoning as InteractionEvent's version)
        -- everything else here is timing/count/identifier metadata, never
        prompt or response content.
        """
        return {
            "call_id": self.call_id, "session_id": self.session_id, "task_id": self.task_id,
            "context_id": self.context_id, "agent_id": self.agent_id, "action_type": self.action_type,
            "triggering_message_id": self.triggering_message_id,
            "input_part_ids": list(self.input_part_ids), "input_artifact_ids": list(self.input_artifact_ids),
            "output_part_ids": list(self.output_part_ids), "output_artifact_ids": list(self.output_artifact_ids),
            "wall_clock_latency_ms": self.wall_clock_latency_ms,
            "llm_called": self.llm_called, "model_name": self.model_name, "model_digest": self.model_digest,
            "prompt_eval_count": self.prompt_eval_count, "eval_count": self.eval_count,
            "prompt_eval_duration": self.prompt_eval_duration, "eval_duration": self.eval_duration,
            "total_duration": self.total_duration, "load_duration": self.load_duration,
            "done_reason": self.done_reason, "retry_count": self.retry_count,
            "error_flag": self.error_flag, "error_type": self.error_type,
            "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed,
            "prompt_config_version": self.prompt_config_version, "agent_config_version": self.agent_config_version,
            "timing_source": self.timing_source, "schema_version": self.schema_version,
        }
