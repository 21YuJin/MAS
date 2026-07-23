"""
[Step 2-3] TravelTask lifecycle: TaskStatus + status-transition validation.

Two validation modes -- same principle applied throughout this package (see
validation.py): an attack can legitimately drive a task through an invalid
status transition, and that IS the observation we want to keep, not discard.
  - strict:      validate_status_transition() raises StatusTransitionError on
                 an invalid transition. For normal workflow development/tests.
  - diagnostic:  never raises; returns whether the transition was valid so the
                 caller can still record the InteractionEvent with
                 status_transition_valid=False rather than losing the record.
"""
import enum


class TaskStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    PLANNING = "planning"
    SEARCHING = "searching"
    WAITING_FOR_INPUT = "waiting_for_input"
    REVISING = "revising"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Every non-terminal status can transition to failed/cancelled at any time --
# added once here rather than repeated in every row below, so a newly added
# active status can't accidentally omit them.
_ACTIVE_STATUSES = {
    TaskStatus.SUBMITTED,
    TaskStatus.PLANNING,
    TaskStatus.SEARCHING,
    TaskStatus.WAITING_FOR_INPUT,
    TaskStatus.REVISING,
    TaskStatus.INTEGRATING,
}

_BASE_TRANSITIONS = {
    TaskStatus.SUBMITTED: {TaskStatus.PLANNING},
    TaskStatus.PLANNING: {TaskStatus.SEARCHING, TaskStatus.WAITING_FOR_INPUT},
    TaskStatus.WAITING_FOR_INPUT: {TaskStatus.PLANNING},
    TaskStatus.SEARCHING: {TaskStatus.REVISING, TaskStatus.INTEGRATING},
    TaskStatus.REVISING: {TaskStatus.SEARCHING, TaskStatus.INTEGRATING},
    TaskStatus.INTEGRATING: {TaskStatus.REVISING, TaskStatus.COMPLETED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}

VALID_TRANSITIONS = {
    before: (afters | {TaskStatus.FAILED, TaskStatus.CANCELLED}
             if before in _ACTIVE_STATUSES else afters)
    for before, afters in _BASE_TRANSITIONS.items()
}


def is_valid_status_transition(before, after) -> bool:
    before = TaskStatus(before)
    after = TaskStatus(after)
    return after in VALID_TRANSITIONS.get(before, set())


class StatusTransitionError(ValueError):
    def __init__(self, before, after):
        super().__init__(f"invalid status transition: {before!r} -> {after!r}")
        self.before = before
        self.after = after


def validate_status_transition(before, after, mode: str = "strict") -> bool:
    """
    mode="strict":      raises StatusTransitionError on an invalid transition.
    mode="diagnostic":  never raises -- returns True/False so the caller can
                        still record the event (status_transition_valid set
                        to this return value) instead of dropping an
                        attack-induced anomalous transition.
    """
    if mode not in ("strict", "diagnostic"):
        raise ValueError(f"unknown validation mode: {mode!r}")
    valid = is_valid_status_transition(before, after)
    if mode == "strict" and not valid:
        raise StatusTransitionError(before, after)
    return valid
