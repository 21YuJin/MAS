"""
[Step 2-6] Cross-object integrity validators.

Two modes throughout, same principle as status.py:
  - strict:      raises ValidationError on the first violation found.
  - diagnostic:  never raises -- returns a list of ValidationIssue, so a
                 caller (e.g. an attack-development run where a corrupted
                 task_id or a broken sender/receiver pair IS the interesting
                 observation) can keep every raw record and still see exactly
                 what's wrong with it, rather than losing it.

Nothing here mutates the objects it validates.
"""
import dataclasses
import datetime as dt
from typing import List, Optional

from .agents import AgentRegistry
from .models import Artifact, InteractionEvent, Message, Part, TravelTask

_VALID_MODES = ("strict", "diagnostic")


class ValidationError(ValueError):
    def __init__(self, code, message, object_id=None, details=None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.object_id = object_id
        self.details = details or {}


@dataclasses.dataclass
class ValidationIssue:
    code: str
    severity: str   # "error" | "warning"
    object_id: Optional[str] = None
    details: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "object_id": self.object_id, "details": dict(self.details)}


def _check_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown validation mode: {mode!r}, expected one of {_VALID_MODES}")


def _report(issues, mode, code, message, object_id=None, details=None, severity="error"):
    if mode == "strict":
        raise ValidationError(code, message, object_id=object_id, details=details)
    issues.append(ValidationIssue(code=code, severity=severity, object_id=object_id, details=details or {}))


def validate_task(task: TravelTask, registry: AgentRegistry, mode: str = "strict") -> List[ValidationIssue]:
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    for agent_id in task.assigned_agent_ids:
        if not registry.contains(agent_id):
            _report(issues, mode, "UNKNOWN_AGENT",
                     f"assigned_agent_ids references unknown agent {agent_id!r}",
                     object_id=task.task_id, details={"agent_id": agent_id})
    return issues


def validate_message(message: Message, registry: AgentRegistry, mode: str = "strict") -> List[ValidationIssue]:
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    valid, reason = registry.validate_sender_receiver(message.sender_id, message.receiver_id)
    if not valid:
        _report(issues, mode, "INVALID_SENDER_RECEIVER", reason, object_id=message.message_id,
                details={"sender_id": message.sender_id, "receiver_id": message.receiver_id})
    return issues


def validate_artifact(artifact: Artifact, registry: AgentRegistry, mode: str = "strict") -> List[ValidationIssue]:
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    if not registry.contains(artifact.producer_id):
        _report(issues, mode, "INVALID_PRODUCER",
                f"producer_id {artifact.producer_id!r} is not a registered agent",
                object_id=artifact.artifact_id, details={"producer_id": artifact.producer_id})
    return issues


def validate_event(event: InteractionEvent, registry: AgentRegistry, mode: str = "strict") -> List[ValidationIssue]:
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    valid, reason = registry.validate_sender_receiver(event.sender_id, event.receiver_id)
    if not valid:
        _report(issues, mode, "INVALID_SENDER_RECEIVER", reason, object_id=event.event_id,
                details={"sender_id": event.sender_id, "receiver_id": event.receiver_id})
    return issues


def validate_context_consistency(task: TravelTask, messages: List[Message], artifacts: List[Artifact],
                                  events: List[InteractionEvent], parts: Optional[List[Part]] = None,
                                  mode: str = "strict") -> List[ValidationIssue]:
    """
    task_id/context_id agreement across every message/artifact/event, plus
    orphan-reference checks (a Message/Artifact pointing at a part_id or
    artifact_id that doesn't exist in the given collections). `parts` is
    optional (not every caller collects Part objects separately) -- when
    omitted, orphan-PART checks are skipped, but orphan-ARTIFACT checks still
    run against `artifacts`.
    """
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    ctx = task.context_id

    def _check_ids(obj, obj_id):
        if obj.task_id != task.task_id:
            _report(issues, mode, "TASK_ID_MISMATCH",
                    f"{obj_id!r} has task_id {obj.task_id!r}, expected {task.task_id!r}",
                    object_id=obj_id, details={"task_id": obj.task_id, "expected": task.task_id})
        if obj.context_id != ctx:
            _report(issues, mode, "CONTEXT_ID_MISMATCH",
                    f"{obj_id!r} has context_id {obj.context_id!r}, expected {ctx!r}",
                    object_id=obj_id, details={"context_id": obj.context_id, "expected": ctx})

    for m in messages:
        _check_ids(m, m.message_id)
    for a in artifacts:
        _check_ids(a, a.artifact_id)
    for e in events:
        _check_ids(e, e.event_id)

    known_artifact_ids = {a.artifact_id for a in artifacts}
    known_part_ids = {p.part_id for p in parts} if parts is not None else None

    for m in messages:
        for aid in m.artifact_ids:
            if aid not in known_artifact_ids:
                _report(issues, mode, "ORPHAN_ARTIFACT_REFERENCE",
                        f"message {m.message_id!r} references unknown artifact_id {aid!r}",
                        object_id=m.message_id, details={"artifact_id": aid})
        if known_part_ids is not None:
            for pid in m.part_ids:
                if pid not in known_part_ids:
                    _report(issues, mode, "ORPHAN_PART_REFERENCE",
                            f"message {m.message_id!r} references unknown part_id {pid!r}",
                            object_id=m.message_id, details={"part_id": pid})

    if known_part_ids is not None:
        for a in artifacts:
            for pid in a.part_ids:
                if pid not in known_part_ids:
                    _report(issues, mode, "ORPHAN_PART_REFERENCE",
                            f"artifact {a.artifact_id!r} references unknown part_id {pid!r}",
                            object_id=a.artifact_id, details={"part_id": pid})

    return issues


def validate_artifact_lineage(artifacts: List[Artifact], mode: str = "strict") -> List[ValidationIssue]:
    """Self-reference is already rejected at Artifact construction time
    (see Artifact.__post_init__) -- this checks the cross-artifact relations
    that need the full collection: every parent/source artifact_id must
    correspond to a known artifact."""
    _check_mode(mode)
    issues: List[ValidationIssue] = []
    known_ids = {a.artifact_id for a in artifacts}
    for a in artifacts:
        for pid in a.parent_artifact_ids:
            if pid not in known_ids:
                _report(issues, mode, "MISSING_PARENT_ARTIFACT",
                        f"artifact {a.artifact_id!r} references unknown parent_artifact_id {pid!r}",
                        object_id=a.artifact_id, details={"parent_artifact_id": pid})
        for sid in a.source_artifact_ids:
            if sid not in known_ids:
                _report(issues, mode, "MISSING_SOURCE_ARTIFACT",
                        f"artifact {a.artifact_id!r} references unknown source_artifact_id {sid!r}",
                        object_id=a.artifact_id, details={"source_artifact_id": sid})
    return issues


def validate_event_sequence(events: List[InteractionEvent], mode: str = "strict") -> List[ValidationIssue]:
    """Duplicate event_index values, and timestamp reversal across
    execution_order (an event starting before the previous one -- in
    event_index order -- ended)."""
    _check_mode(mode)
    issues: List[ValidationIssue] = []

    seen_indices = {}
    for e in events:
        if e.event_index in seen_indices:
            _report(issues, mode, "DUPLICATE_EVENT_INDEX",
                    f"event_index {e.event_index} used by both {seen_indices[e.event_index]!r} and {e.event_id!r}",
                    object_id=e.event_id, details={"event_index": e.event_index})
        else:
            seen_indices[e.event_index] = e.event_id

    ordered = sorted(events, key=lambda e: e.event_index)
    prev_end = None
    for e in ordered:
        if prev_end is not None and e.start_timestamp:
            if dt.datetime.fromisoformat(e.start_timestamp) < dt.datetime.fromisoformat(prev_end):
                _report(issues, mode, "TIMESTAMP_REVERSAL",
                        f"event {e.event_id!r} starts before the previous event (by event_index) ended",
                        object_id=e.event_id,
                        details={"start_timestamp": e.start_timestamp, "previous_end_timestamp": prev_end})
        if e.end_timestamp:
            prev_end = e.end_timestamp
    return issues
