"""
[Step 3-2] Source-controlled TravelTask fixtures -- loads
data/travel_a2a/development/tasks/normal_tasks.json and builds (TravelTask,
expected_branches) pairs.

[Step 6.5-1] These 6 fixtures are DEVELOPMENT fixtures only (smoke test /
workflow regression / evaluator unit tests / attack development) -- they are
never mixed with the formal_workload/ task instances built in Step 6.5
onward. Kept at data/travel_a2a/development/ specifically so a future formal
dataset script can never accidentally read/write this path (CLAUDE.md's
existing-path-isolation principle, applied one level down within
data/travel_a2a/ itself).

expected_branches is diagnostic-only ground truth for tests (which
conditional collaboration branch, Step 3-6, a fixture is DESIGNED to trigger
via its budget/dates/content-fixture combination) -- it is NOT a
TravelRequest/TravelTask field and never reaches the object models
themselves, so it can never leak into anything workflow_policy.py inspects.

task_id/context_id are derived deterministically from task_fixture_id
(f"task_{fixture_id}" / f"ctx_{fixture_id}") rather than from a random/counter
ID factory, so the same fixture always produces the same IDs across runs --
required for the deterministic-repeat-run test (Step 3 unit test #3).
"""
import json
import os
from typing import List, Tuple

from .models import TravelRequest, TravelTask
from .status import TaskStatus

DEFAULT_TASKS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "data", "travel_a2a", "development", "tasks", "normal_tasks.json")

FIXTURE_CREATED_AT = "2026-08-15T00:00:00+00:00"


def load_task_fixture_dicts(path: str = DEFAULT_TASKS_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_travel_task(fixture: dict, task_id: str, context_id: str,
                       created_at: str = FIXTURE_CREATED_AT) -> TravelTask:
    request = TravelRequest(
        origin=fixture["origin"], destination=fixture["destination"],
        departure_date=fixture["departure_date"], return_date=fixture["return_date"],
        travelers=fixture["travelers"], budget_amount=fixture["budget_amount"],
        budget_currency=fixture["budget_currency"], target_currency=fixture["target_currency"],
        flight_preferences=fixture.get("flight_preferences", {}),
        hotel_preferences=fixture.get("hotel_preferences", {}),
        activity_preferences=fixture.get("activity_preferences", {}),
        required_services=fixture["required_services"],
        task_category=fixture["task_category"], difficulty=fixture["difficulty"],
    )
    return TravelTask(
        task_id=task_id, context_id=context_id, request=request,
        status=TaskStatus.SUBMITTED, condition="normal",
        injection_present=False, attack_id=None,
        created_at=created_at, updated_at=created_at,
        provenance={"task_fixture_id": fixture["task_fixture_id"]},
    )


def load_travel_tasks(path: str = DEFAULT_TASKS_PATH) -> List[Tuple[TravelTask, List[str]]]:
    """Returns one (TravelTask, expected_branches) pair per fixture, in
    fixture-file order."""
    out = []
    for fx in load_task_fixture_dicts(path):
        fixture_id = fx["task_fixture_id"]
        task = build_travel_task(fx, task_id=f"task_{fixture_id}", context_id=f"ctx_{fixture_id}")
        out.append((task, list(fx.get("expected_branches", []))))
    return out
