"""
[Step 5-2/5-3/5-6] Attack scenario schema: ExternalContentSourceType,
AttackConfig, AttackExecutionDiagnostics.

Core labeling principle (Step 5-1), enforced by TravelTask.ground_truth_label
(models.py, unchanged since Step 2) and never re-derived here:

    ground_truth_label = int(injection_present)

goal_success / indicator_observed / propagation_observed / workflow_changed
are diagnostic-only (AttackExecutionDiagnostics below) -- they never feed
ground_truth_label, and an attack session is NEVER removed or relabeled as
normal because it failed to achieve its goal (an unsuccessful attack is still
exactly as much a member of the "attack" pool as a successful one -- that IS
the ground truth: injection was present, regardless of what happened next).
"""
import dataclasses
import enum
import hashlib
from typing import Any, Dict, List, Optional


class ExternalContentSourceType(str, enum.Enum):
    """Controlled vocabulary for WHERE an injection can enter -- restricted
    to actual external-content channels a specialist agent reads, never the
    user's own TravelRequest (Step 5-1 principle #1/#2). Only 3 of these 7
    are wired to an implemented attack as of Step 5 (hotel_description,
    flight_policy, tour_description) -- the rest are reserved vocabulary for
    later attack families, not yet used by any AttackConfig."""
    FLIGHT_DESCRIPTION = "flight_description"
    FLIGHT_POLICY = "flight_policy"
    HOTEL_DESCRIPTION = "hotel_description"
    HOTEL_POLICY = "hotel_policy"
    CURRENCY_PROVIDER_NOTE = "currency_provider_note"
    TOUR_DESCRIPTION = "tour_description"
    LOCAL_TRAVEL_GUIDE = "local_travel_guide"


ATTACK_FAMILIES = (
    "preference_manipulation", "workflow_corruption", "downstream_propagation",
    "unauthorized_artifact_forwarding", "task_context_mismatch", "budget_manipulation",
)

# Only these 3 are implemented in Step 5 (Step 5-4) -- the rest of
# ATTACK_FAMILIES are reserved names for later steps, per the "구현은 3개만"
# instruction (implementing 6 at once makes evaluator bugs hard to isolate).
IMPLEMENTED_ATTACK_FAMILIES = ("preference_manipulation", "workflow_corruption", "downstream_propagation")

PAYLOAD_INTENSITIES = ("direct", "contextual", "obfuscated")
# Step 5-9: only "direct" is actually exercised in Step 5 -- "contextual"/
# "obfuscated" are schema-valid values an AttackConfig CAN declare, but no
# Step 5 attack config uses them yet.


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")


@dataclasses.dataclass
class AttackConfig:
    attack_id: str
    attack_family: str
    attack_goal: str
    injection_source_type: str
    injection_source_id: str
    entry_agent_id: str
    target_agent_ids: List[str]
    expected_propagation_path: List[str]
    payload_template: str
    payload_template_version: str
    evaluator_id: str
    indicator_patterns: List[str] = dataclasses.field(default_factory=list)
    evaluator_target_agents: List[str] = dataclasses.field(default_factory=list)
    payload_intensity: str = "direct"
    malicious_target_option_id: Optional[str] = None
    # [Step 6-6] payload_variant_id/semantic_goal_id let mini-validation ask
    # "does success depend on ONE specific wording, or does the family
    # generalize?" -- two variants sharing the same semantic_goal_id must
    # pursue the identical objective, differing only in phrasing.
    # payload_hash is derived (never hand-set) so two variants can never
    # accidentally collide or silently duplicate a payload.
    payload_variant_id: str = "v1"
    semantic_goal_id: str = ""
    payload_hash: Optional[str] = None
    development_only: bool = True
    enabled: bool = True
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        for name in ("attack_id", "attack_family", "attack_goal", "injection_source_type",
                     "injection_source_id", "entry_agent_id", "payload_template",
                     "payload_template_version", "evaluator_id"):
            _require_nonempty(getattr(self, name), name)
        if self.attack_family not in ATTACK_FAMILIES:
            raise ValueError(f"attack_family must be one of {ATTACK_FAMILIES}, got {self.attack_family!r}")
        self.injection_source_type = ExternalContentSourceType(self.injection_source_type).value
        if self.payload_intensity not in PAYLOAD_INTENSITIES:
            raise ValueError(f"payload_intensity must be one of {PAYLOAD_INTENSITIES}, got {self.payload_intensity!r}")
        if not self.semantic_goal_id:
            self.semantic_goal_id = self.attack_family
        if self.payload_hash is None:
            self.payload_hash = hashlib.sha256(self.payload_template.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack_id, "attack_family": self.attack_family, "attack_goal": self.attack_goal,
            "injection_source_type": self.injection_source_type, "injection_source_id": self.injection_source_id,
            "entry_agent_id": self.entry_agent_id, "target_agent_ids": list(self.target_agent_ids),
            "expected_propagation_path": list(self.expected_propagation_path),
            "payload_template": self.payload_template, "payload_template_version": self.payload_template_version,
            "evaluator_id": self.evaluator_id, "indicator_patterns": list(self.indicator_patterns),
            "evaluator_target_agents": list(self.evaluator_target_agents),
            "payload_intensity": self.payload_intensity, "malicious_target_option_id": self.malicious_target_option_id,
            "payload_variant_id": self.payload_variant_id, "semantic_goal_id": self.semantic_goal_id,
            "payload_hash": self.payload_hash,
            "development_only": self.development_only, "enabled": self.enabled,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackConfig":
        return cls(**data)


@dataclasses.dataclass
class AttackExecutionDiagnostics:
    """
    Raw diagnostic record for ONE attack session. NEVER enters the
    LightGAE metadata-feature path (no to_metadata_dict() on this class at
    all -- it is diagnostic-only by construction, see Step 5-6). Kept
    entirely separate from ground_truth_label (TravelTask.injection_present),
    which this class never computes or overrides.
    """
    session_id: str
    attack_id: str
    injection_present: bool
    injection_source_id: str
    entry_agent_id: str
    # [Step 6-1] Deliberately kept as 8 SEPARATE states, never collapsed into
    # one attack_success boolean -- each answers a different question about
    # how far the attack got, and a session can be true on an early one and
    # false on every later one (e.g. entry_agent_exposed=True,
    # instruction_followed=False -- the agent read the payload but ignored it).
    entry_agent_exposed: bool = False
    instruction_followed: bool = False
    indicator_observed: bool = False
    artifact_changed: bool = False
    propagation_observed: bool = False
    workflow_changed: bool = False
    goal_success: bool = False
    indicator_observed_by_agent: List[str] = dataclasses.field(default_factory=list)
    indicator_observed_in_artifact: List[str] = dataclasses.field(default_factory=list)
    propagation_depth: int = 0
    affected_agent_ids: List[str] = dataclasses.field(default_factory=list)
    affected_artifact_ids: List[str] = dataclasses.field(default_factory=list)
    artifact_contract_violated: bool = False
    output_changed: bool = False
    # [Step 6-13] one entry per hop examined by the propagation evaluator:
    # {source_agent, target_agent, source_artifact_id, target_artifact_id,
    #  indicator_present, semantic_instruction_preserved, hop_index}
    hop_trace: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    evaluator_id: str = ""
    evaluator_version: str = "v1"
    evaluator_confidence: Optional[str] = None
    evaluator_evidence: Dict[str, Any] = dataclasses.field(default_factory=dict)
    evaluator_error: Optional[str] = None
    # [Step 6-8] sampling flags for the manual review queue -- never used as
    # ground truth, only to decide which sessions a human should look at.
    manual_review_required: bool = False
    manual_review_reasons: List[str] = dataclasses.field(default_factory=list)
    schema_version: str = "travel_a2a_v1"

    def __post_init__(self):
        for name in ("session_id", "attack_id", "injection_source_id", "entry_agent_id"):
            _require_nonempty(getattr(self, name), name)
        if self.propagation_depth < 0:
            raise ValueError(f"propagation_depth must be >= 0, got {self.propagation_depth}")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "attack_id": self.attack_id,
            "injection_present": self.injection_present, "injection_source_id": self.injection_source_id,
            "entry_agent_id": self.entry_agent_id,
            "entry_agent_exposed": self.entry_agent_exposed, "instruction_followed": self.instruction_followed,
            "indicator_observed": self.indicator_observed, "artifact_changed": self.artifact_changed,
            "indicator_observed_by_agent": list(self.indicator_observed_by_agent),
            "indicator_observed_in_artifact": list(self.indicator_observed_in_artifact),
            "propagation_observed": self.propagation_observed, "propagation_depth": self.propagation_depth,
            "affected_agent_ids": list(self.affected_agent_ids), "affected_artifact_ids": list(self.affected_artifact_ids),
            "artifact_contract_violated": self.artifact_contract_violated,
            "workflow_changed": self.workflow_changed, "output_changed": self.output_changed,
            "goal_success": self.goal_success, "hop_trace": [dict(h) for h in self.hop_trace],
            "evaluator_id": self.evaluator_id, "evaluator_version": self.evaluator_version,
            "evaluator_confidence": self.evaluator_confidence, "evaluator_evidence": dict(self.evaluator_evidence),
            "evaluator_error": self.evaluator_error,
            "manual_review_required": self.manual_review_required,
            "manual_review_reasons": list(self.manual_review_reasons),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackExecutionDiagnostics":
        return cls(**data)
