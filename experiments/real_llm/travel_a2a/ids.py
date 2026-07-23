"""
[Step 2-5] ID generation utilities for travel_a2a objects.

Each `new_*_id()` accepts an optional zero-arg `factory` callable that
returns just the suffix (defaults to a random UUID4-derived one), so callers
-- tests in particular -- can inject a deterministic sequence instead of a
fresh UUID every run. Prefixes are chosen so an ID's kind is recognizable at
a glance in logs/dumps without looking up a schema.
"""
import uuid


def _default_suffix() -> str:
    return uuid.uuid4().hex[:12]


def _make_id(prefix: str, factory=None) -> str:
    suffix = (factory or _default_suffix)()
    return f"{prefix}{suffix}"


def new_session_id(factory=None) -> str:
    return _make_id("session_", factory)


def new_task_id(factory=None) -> str:
    return _make_id("task_", factory)


def new_context_id(factory=None) -> str:
    return _make_id("ctx_", factory)


def new_message_id(factory=None) -> str:
    return _make_id("msg_", factory)


def new_part_id(factory=None) -> str:
    return _make_id("part_", factory)


def new_artifact_id(factory=None) -> str:
    return _make_id("artifact_", factory)


def new_event_id(factory=None) -> str:
    return _make_id("event_", factory)


def new_call_id(factory=None) -> str:
    return _make_id("call_", factory)


class DeterministicIdFactory:
    """
    Injectable ID source for tests: one independent zero-padded counter per
    ID kind, producing stable sequences (task_000000, task_000001, ...)
    instead of random UUID suffixes, so snapshot-style test assertions don't
    need to tolerate a fresh random ID on every run.
    """

    def __init__(self):
        self._counters = {}

    def _next(self, kind: str) -> str:
        n = self._counters.get(kind, 0)
        self._counters[kind] = n + 1
        return f"{n:06d}"

    def session_id(self) -> str:
        return new_session_id(factory=lambda: self._next("session"))

    def task_id(self) -> str:
        return new_task_id(factory=lambda: self._next("task"))

    def context_id(self) -> str:
        return new_context_id(factory=lambda: self._next("ctx"))

    def message_id(self) -> str:
        return new_message_id(factory=lambda: self._next("msg"))

    def part_id(self) -> str:
        return new_part_id(factory=lambda: self._next("part"))

    def artifact_id(self) -> str:
        return new_artifact_id(factory=lambda: self._next("artifact"))

    def event_id(self) -> str:
        return new_event_id(factory=lambda: self._next("event"))

    def call_id(self) -> str:
        return new_call_id(factory=lambda: self._next("call"))
