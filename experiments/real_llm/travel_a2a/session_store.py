"""
[Step 3-9/3-10] Save/reload one mock session as a directory of files, with
RAW records (task.json/messages.jsonl/parts.jsonl/artifacts.jsonl/
interaction_events.jsonl) and metadata-only (content-free) records
(metadata/*_metadata.jsonl) kept in separate files -- the metadata/ files
must never contain a FORBIDDEN_METADATA_KEYS field (models.py), checked here
by validate_no_forbidden_metadata_fields() (also exercised directly by the
Step 3 unit tests) every time save_session() runs.

Directory layout:
    outputs/travel_a2a/mock_sessions/<session_id>/
    |-- task.json
    |-- request.json
    |-- messages.jsonl
    |-- parts.jsonl
    |-- artifacts.jsonl
    |-- interaction_events.jsonl
    |-- session_result.json          (optional)
    |-- validation_report.json       (optional)
    `-- metadata/
        |-- messages_metadata.jsonl
        |-- parts_metadata.jsonl
        |-- artifacts_metadata.jsonl
        `-- events_metadata.jsonl
"""
import json
import os
from typing import List, Optional

from .models import (
    Artifact, FORBIDDEN_METADATA_KEYS, InteractionEvent, Message, Part, TravelTask,
)

DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "travel_a2a", "mock_sessions")


def session_dir_for(session_id: str, output_root: str = DEFAULT_OUTPUT_ROOT) -> str:
    return os.path.join(output_root, session_id)


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_session(session_id: str, task: TravelTask, messages: List[Message], parts: List[Part],
                  artifacts: List[Artifact], events: List[InteractionEvent],
                  session_result: Optional[dict] = None, validation_report: Optional[dict] = None,
                  output_root: str = DEFAULT_OUTPUT_ROOT) -> str:
    session_dir = session_dir_for(session_id, output_root)
    metadata_dir = os.path.join(session_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    _write_json(os.path.join(session_dir, "task.json"), task.to_dict())
    _write_json(os.path.join(session_dir, "request.json"), task.request.to_dict())
    _write_jsonl(os.path.join(session_dir, "messages.jsonl"), [m.to_dict() for m in messages])
    _write_jsonl(os.path.join(session_dir, "parts.jsonl"), [p.to_dict() for p in parts])
    _write_jsonl(os.path.join(session_dir, "artifacts.jsonl"), [a.to_dict() for a in artifacts])
    _write_jsonl(os.path.join(session_dir, "interaction_events.jsonl"), [e.to_dict() for e in events])
    if session_result is not None:
        _write_json(os.path.join(session_dir, "session_result.json"), session_result)
    if validation_report is not None:
        _write_json(os.path.join(session_dir, "validation_report.json"), validation_report)

    _write_jsonl(os.path.join(metadata_dir, "messages_metadata.jsonl"), [m.to_metadata_dict() for m in messages])
    _write_jsonl(os.path.join(metadata_dir, "parts_metadata.jsonl"), [p.to_metadata_dict() for p in parts])
    _write_jsonl(os.path.join(metadata_dir, "artifacts_metadata.jsonl"), [a.to_metadata_dict() for a in artifacts])
    _write_jsonl(os.path.join(metadata_dir, "events_metadata.jsonl"), [e.to_metadata_dict() for e in events])

    validate_no_forbidden_metadata_fields(session_dir)
    return session_dir


def validate_no_forbidden_metadata_fields(session_dir: str) -> None:
    """Re-reads every metadata/*.jsonl file just written and asserts none of
    them contain a FORBIDDEN_METADATA_KEYS field -- a save-time guard, not
    just a unit-test-time one, so this check runs on every real session
    written, not only on whatever the test suite happens to construct."""
    metadata_dir = os.path.join(session_dir, "metadata")
    for fname in ("messages_metadata.jsonl", "parts_metadata.jsonl",
                  "artifacts_metadata.jsonl", "events_metadata.jsonl"):
        for record in _read_jsonl(os.path.join(metadata_dir, fname)):
            leaked = FORBIDDEN_METADATA_KEYS & set(record.keys())
            if leaked:
                raise AssertionError(f"{fname} contains forbidden metadata key(s): {leaked}")


def load_session(session_id: str, output_root: str = DEFAULT_OUTPUT_ROOT) -> dict:
    """Reconstructs task/messages/parts/artifacts/events from the RAW files
    (not metadata/) -- the round-trip counterpart to save_session(). Returns
    a plain dict rather than a dataclass since callers generally want to
    unpack only some of these collections."""
    session_dir = session_dir_for(session_id, output_root)
    with open(os.path.join(session_dir, "task.json"), encoding="utf-8") as f:
        task = TravelTask.from_dict(json.load(f))
    messages = [Message.from_dict(d) for d in _read_jsonl(os.path.join(session_dir, "messages.jsonl"))]
    parts = [Part.from_dict(d) for d in _read_jsonl(os.path.join(session_dir, "parts.jsonl"))]
    artifacts = [Artifact.from_dict(d) for d in _read_jsonl(os.path.join(session_dir, "artifacts.jsonl"))]
    events = [InteractionEvent.from_dict(d) for d in _read_jsonl(os.path.join(session_dir, "interaction_events.jsonl"))]
    return {"task": task, "messages": messages, "parts": parts, "artifacts": artifacts, "events": events}
