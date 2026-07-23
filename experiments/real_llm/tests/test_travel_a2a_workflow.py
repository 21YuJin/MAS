"""
[Step 3] Unit tests for the deterministic mock travel workflow: fixtures,
content repository, workflow policy branches, session save/reload, and
structural diversity across the 6 task fixtures. No Ollama calls -- this is
a pure object/state-machine test, per the Step 3 scope boundary.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_workflow.py
"""
import copy
import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.content_repository import load_content_repository  # noqa: E402
from travel_a2a.fixtures import load_travel_tasks  # noqa: E402
from travel_a2a.mock_runner import DeterministicClock, MockTravelSessionRunner, run_mock_workflow  # noqa: E402
from travel_a2a.models import ArtifactType, FORBIDDEN_METADATA_KEYS, InteractionType  # noqa: E402
from travel_a2a.session_store import DEFAULT_OUTPUT_ROOT, load_session, save_session  # noqa: E402
from travel_a2a.status import TaskStatus  # noqa: E402
from travel_a2a.validation import (  # noqa: E402
    ValidationError, validate_artifact, validate_artifact_lineage,
    validate_context_consistency, validate_event, validate_event_sequence, validate_message,
)
from travel_a2a.agents import build_default_registry  # noqa: E402
from travel_a2a.ids import DeterministicIdFactory  # noqa: E402


def _run(fixture_index):
    """Fresh copy of the fixture task + fresh id_factory/clock every call,
    so tests never share mutated state (task.status mutates in place)."""
    tasks = load_travel_tasks()
    task, expected_branches = tasks[fixture_index]
    task = copy.deepcopy(task)
    repo = load_content_repository()
    result = run_mock_workflow(task, repo, id_factory=DeterministicIdFactory(), clock=DeterministicClock())
    return task, expected_branches, result


FIXTURE_INDEX = {
    "easy_basic_tokyo": 0, "easy_business_singapore": 1, "medium_budget_osaka": 2,
    "medium_family_bangkok": 3, "hard_activity_paris": 4, "hard_multi_constraint_london": 5,
}


class TestFixturesAndContent(unittest.TestCase):
    def test_01_load_six_task_fixtures(self):
        tasks = load_travel_tasks()
        self.assertEqual(len(tasks), 6)
        fixture_ids = {t.provenance["task_fixture_id"] for t, _ in tasks}
        self.assertEqual(fixture_ids, set(FIXTURE_INDEX.keys()))

    def test_02_load_content_repository(self):
        repo = load_content_repository()
        self.assertTrue(repo.flights_for("Tokyo"))
        self.assertTrue(repo.hotels_for("Osaka"))
        self.assertTrue(repo.tours_for("Paris"))
        self.assertAlmostEqual(repo.currency_rate("KRW", "JPY"), 0.11)
        # designed to be empty -- see mock_agents.py / workflow_policy.py
        # integration_revision branch docstrings
        self.assertEqual(repo.tours_for_in_range("London", "2026-10-10", "2026-10-15"), [])


class TestDeterminism(unittest.TestCase):
    def test_03_repeated_run_is_deterministic(self):
        for name, idx in FIXTURE_INDEX.items():
            _, _, r1 = _run(idx)
            _, _, r2 = _run(idx)
            self.assertEqual([m.to_dict() for m in r1.messages], [m.to_dict() for m in r2.messages], name)
            self.assertEqual([e.to_dict() for e in r1.events], [e.to_dict() for e in r2.events], name)


class TestBaseWorkflow(unittest.TestCase):
    def test_04_base_workflow_completes(self):
        task, expected, result = _run(FIXTURE_INDEX["easy_basic_tokyo"])
        self.assertEqual(expected, [])
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(result.status_transition_issues, [])
        interaction_types = [m.interaction_type for m in result.messages]
        self.assertEqual(interaction_types[0], InteractionType.TASK_SUBMISSION)
        self.assertEqual(interaction_types[-1], InteractionType.TASK_COMPLETION)


class TestConditionalBranches(unittest.TestCase):
    def test_05_budget_conflict_branch(self):
        task, expected, result = _run(FIXTURE_INDEX["medium_budget_osaka"])
        self.assertIn("budget_conflict", expected)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        revisions = [m for m in result.messages if m.interaction_type == InteractionType.REVISION_REQUEST
                     and m.receiver_id == "hotel_agent"]
        self.assertEqual(len(revisions), 1)
        hotel_artifacts = [a for a in result.artifacts if a.artifact_type == ArtifactType.HOTEL_OPTIONS]
        self.assertTrue(any(a.version == 2 for a in hotel_artifacts))
        clarify = [m for m in result.messages if m.sender_id == "hotel_agent" and m.receiver_id == "currency_agent"]
        self.assertEqual(len(clarify), 1)

    def test_06_schedule_conflict_branch(self):
        task, expected, result = _run(FIXTURE_INDEX["hard_activity_paris"])
        self.assertIn("schedule_conflict", expected)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        clarify = [m for m in result.messages if m.sender_id == "tours_agent" and m.receiver_id == "flight_agent"
                   and m.interaction_type == InteractionType.CLARIFICATION_REQUEST]
        self.assertEqual(len(clarify), 1)
        tour_artifacts = [a for a in result.artifacts if a.artifact_type == ArtifactType.TOUR_OPTIONS]
        v2 = [a for a in tour_artifacts if a.version == 2]
        self.assertEqual(len(v2), 1)
        self.assertLess(v2[0].record_count, max(a.record_count for a in tour_artifacts if a.version == 1))

    def test_07_client_clarification_branch(self):
        task, expected, result = _run(FIXTURE_INDEX["medium_family_bangkok"])
        self.assertIn("client_clarification", expected)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        asks = [m for m in result.messages if m.sender_id == "travel_coordinator" and m.receiver_id == "client"
                and m.interaction_type == InteractionType.CLARIFICATION_REQUEST]
        answers = [m for m in result.messages if m.sender_id == "client" and m.receiver_id == "travel_coordinator"
                   and m.interaction_type == InteractionType.CLARIFICATION_RESPONSE]
        self.assertEqual(len(asks), 1)
        self.assertEqual(len(answers), 1)

    def test_08_integration_revision_branch(self):
        task, expected, result = _run(FIXTURE_INDEX["hard_multi_constraint_london"])
        self.assertIn("integration_revision", expected)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        revisions = [m for m in result.messages if m.interaction_type == InteractionType.REVISION_REQUEST
                     and m.receiver_id == "tours_agent"]
        self.assertEqual(len(revisions), 1)
        tour_artifacts = sorted((a for a in result.artifacts if a.artifact_type == ArtifactType.TOUR_OPTIONS),
                                 key=lambda a: a.version)
        self.assertEqual(tour_artifacts[0].record_count, 0)
        self.assertGreater(tour_artifacts[-1].record_count, 0)


class TestArtifactAndEventIntegrity(unittest.TestCase):
    def test_09_artifact_version_increases(self):
        _, _, result = _run(FIXTURE_INDEX["medium_budget_osaka"])
        versions = sorted(a.version for a in result.artifacts if a.artifact_type == ArtifactType.HOTEL_OPTIONS)
        self.assertEqual(versions, [1, 2])

    def test_10_parent_source_artifact_lineage(self):
        _, _, result = _run(FIXTURE_INDEX["easy_basic_tokyo"])
        issues = validate_artifact_lineage(result.artifacts, mode="diagnostic")
        self.assertEqual(issues, [])
        final_plan = next(a for a in result.artifacts if a.artifact_type == ArtifactType.FINAL_TRAVEL_PLAN)
        integrated = next(a for a in result.artifacts if a.artifact_type == ArtifactType.INTEGRATED_ITINERARY)
        self.assertEqual(final_plan.parent_artifact_ids, [integrated.artifact_id])
        self.assertIn(integrated.artifact_id, {a.artifact_id for a in result.artifacts})

    def test_11_message_event_sender_receiver_match(self):
        _, _, result = _run(FIXTURE_INDEX["hard_multi_constraint_london"])
        messages_by_id = {m.message_id: m for m in result.messages}
        for e in result.events:
            self.assertEqual(e.sender_id, messages_by_id[e.message_id].sender_id)
            self.assertEqual(e.receiver_id, messages_by_id[e.message_id].receiver_id)

    def test_12_event_index_contiguous(self):
        _, _, result = _run(FIXTURE_INDEX["medium_budget_osaka"])
        indices = [e.event_index for e in result.events]
        self.assertEqual(indices, list(range(len(result.events))))
        issues = validate_event_sequence(result.events, mode="diagnostic")
        self.assertEqual(issues, [])

    def test_13_strict_validation_zero_errors(self):
        registry = build_default_registry()
        for name, idx in FIXTURE_INDEX.items():
            task, _, result = _run(idx)
            for m in result.messages:
                validate_message(m, registry, mode="strict")
            for a in result.artifacts:
                validate_artifact(a, registry, mode="strict")
            for e in result.events:
                validate_event(e, registry, mode="strict")
            validate_context_consistency(task, result.messages, result.artifacts, result.events,
                                          parts=result.parts, mode="strict")
            validate_artifact_lineage(result.artifacts, mode="strict")
            validate_event_sequence(result.events, mode="strict")

    def test_14_metadata_views_have_no_forbidden_fields(self):
        for name, idx in FIXTURE_INDEX.items():
            _, _, result = _run(idx)
            for m in result.messages:
                self.assertEqual(FORBIDDEN_METADATA_KEYS & set(m.to_metadata_dict().keys()), set(), name)
            for p in result.parts:
                self.assertEqual(FORBIDDEN_METADATA_KEYS & set(p.to_metadata_dict().keys()), set(), name)
            for a in result.artifacts:
                self.assertEqual(FORBIDDEN_METADATA_KEYS & set(a.to_metadata_dict().keys()), set(), name)
            for e in result.events:
                self.assertEqual(FORBIDDEN_METADATA_KEYS & set(e.to_metadata_dict().keys()), set(), name)


class TestStructuralDiversity(unittest.TestCase):
    def test_15_at_least_three_distinct_event_patterns(self):
        patterns = set()
        for name, idx in FIXTURE_INDEX.items():
            _, _, result = _run(idx)
            pattern = tuple(e.interaction_type.value for e in result.events)
            patterns.add(pattern)
        self.assertGreaterEqual(len(patterns), 3, f"only {len(patterns)} distinct pattern(s): {patterns}")


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self._dirs_to_clean = []

    def tearDown(self):
        for d in self._dirs_to_clean:
            if os.path.isdir(d):
                shutil.rmtree(d)

    def test_16_session_run_result_json_serializable(self):
        tasks = load_travel_tasks()
        task, _ = tasks[FIXTURE_INDEX["easy_basic_tokyo"]]
        task = copy.deepcopy(task)
        runner = MockTravelSessionRunner(load_content_repository())
        session_id = "unittest_easy_basic_tokyo"
        session_result = runner.run(task, "normal", session_id=session_id,
                                     id_factory=DeterministicIdFactory(), clock=DeterministicClock())
        payload = json.dumps({
            "session_id": session_result.session_id, "task_id": session_result.task_id,
            "context_id": session_result.context_id, "agent_call_records": session_result.agent_call_records,
            "messages": session_result.messages, "parts": session_result.parts,
            "artifacts": session_result.artifacts, "interaction_events": session_result.interaction_events,
            "final_output": session_result.final_output, "diagnostic_labels": session_result.diagnostic_labels,
            "errors": session_result.errors,
        })
        self.assertTrue(len(payload) > 0)
        self.assertTrue(len(session_result.agent_call_records) > 0)
        self.assertIsNotNone(session_result.final_output)

    def test_17_save_and_reload_session(self):
        tasks = load_travel_tasks()
        task, _ = tasks[FIXTURE_INDEX["medium_budget_osaka"]]
        task = copy.deepcopy(task)
        repo = load_content_repository()
        session_id = "unittest_medium_budget_osaka"
        result = run_mock_workflow(task, repo, id_factory=DeterministicIdFactory(), clock=DeterministicClock(),
                                    session_id=session_id)
        session_dir = save_session(session_id, task, result.messages, result.parts, result.artifacts, result.events)
        self._dirs_to_clean.append(session_dir)

        reloaded = load_session(session_id)
        self.assertEqual(len(reloaded["messages"]), len(result.messages))
        self.assertEqual(len(reloaded["parts"]), len(result.parts))
        self.assertEqual(len(reloaded["artifacts"]), len(result.artifacts))
        self.assertEqual(len(reloaded["events"]), len(result.events))
        self.assertEqual(reloaded["task"].task_id, task.task_id)
        self.assertEqual(reloaded["task"].status, task.status)
        # every part_id/artifact_id referenced by a reloaded message must
        # resolve to a reloaded part/artifact
        part_ids = {p.part_id for p in reloaded["parts"]}
        artifact_ids = {a.artifact_id for a in reloaded["artifacts"]}
        for m in reloaded["messages"]:
            for pid in m.part_ids:
                self.assertIn(pid, part_ids)
            for aid in m.artifact_ids:
                self.assertIn(aid, artifact_ids)


def print_structural_diversity_summary():
    """[Step 3-11] Not a unit test -- prints the per-task summary table the
    Step 3 instruction asks for, and checks the minimum passing criteria."""
    print("\n" + "=" * 100)
    print("  STRUCTURAL DIVERSITY SUMMARY (Step 3-11)")
    print("=" * 100)

    rows = []
    all_patterns = set()
    any_budget_revision = False
    any_schedule_clarification = False
    any_client_clarification = False
    any_artifact_v2 = False
    all_completed = True
    any_validation_error = False

    for name, idx in FIXTURE_INDEX.items():
        task, expected, result = _run(idx)
        active_agents = sorted({m.sender_id for m in result.messages} | {m.receiver_id for m in result.messages})
        directed_pairs = [(m.sender_id, m.receiver_id) for m in result.messages]
        unique_pairs = set(directed_pairs)
        repeated_pairs = len(directed_pairs) - len(unique_pairs)
        clar_req = sum(1 for m in result.messages if m.interaction_type == InteractionType.CLARIFICATION_REQUEST)
        clar_resp = sum(1 for m in result.messages if m.interaction_type == InteractionType.CLARIFICATION_RESPONSE)
        rev_req = sum(1 for m in result.messages if m.interaction_type == InteractionType.REVISION_REQUEST)
        artifact_versions = sorted({a.version for a in result.artifacts})

        pattern = tuple(e.interaction_type.value for e in result.events)
        all_patterns.add(pattern)
        if any(m.receiver_id == "hotel_agent" and m.interaction_type == InteractionType.REVISION_REQUEST for m in result.messages):
            any_budget_revision = True
        if any(m.sender_id == "tours_agent" and m.receiver_id == "flight_agent" for m in result.messages):
            any_schedule_clarification = True
        if any(m.sender_id == "travel_coordinator" and m.receiver_id == "client"
               and m.interaction_type == InteractionType.CLARIFICATION_REQUEST for m in result.messages):
            any_client_clarification = True
        if 2 in artifact_versions:
            any_artifact_v2 = True
        if task.status != TaskStatus.COMPLETED:
            all_completed = False

        registry = build_default_registry()
        try:
            for m in result.messages:
                validate_message(m, registry, mode="strict")
            validate_event_sequence(result.events, mode="strict")
            validate_artifact_lineage(result.artifacts, mode="strict")
        except ValidationError:
            any_validation_error = True

        rows.append({
            "task_fixture_id": name, "active_agents": active_agents, "n_events": len(result.events),
            "unique_directed_pairs": len(unique_pairs), "repeated_pairs": repeated_pairs,
            "clarification_request": clar_req, "clarification_response": clar_resp,
            "revision_request": rev_req, "n_artifacts": len(result.artifacts),
            "artifact_versions": artifact_versions, "status": task.status.value,
        })

    for r in rows:
        print(f"  {r['task_fixture_id']:<28} events={r['n_events']:>3}  unique_pairs={r['unique_directed_pairs']:>2}  "
              f"repeated_pairs={r['repeated_pairs']}  clar_req={r['clarification_request']}  "
              f"clar_resp={r['clarification_response']}  rev_req={r['revision_request']}  "
              f"artifacts={r['n_artifacts']}  versions={r['artifact_versions']}  status={r['status']}")

    event_counts = {r["task_fixture_id"]: r["n_events"] for r in rows}
    print(f"\n  distinct event patterns: {len(all_patterns)}")
    print(f"  event counts by task: {event_counts}")
    print(f"  all completed: {all_completed}   any validation error: {any_validation_error}")
    print(f"  any budget revision: {any_budget_revision}   any schedule clarification: {any_schedule_clarification}")
    print(f"  any client clarification: {any_client_clarification}   any artifact v2: {any_artifact_v2}")

    checks = {
        "6 sessions completed": all_completed,
        "0 validation errors": not any_validation_error,
        ">=3 distinct event patterns": len(all_patterns) >= 3,
        "event counts not all identical": len(set(event_counts.values())) > 1,
        ">=1 budget revision": any_budget_revision,
        ">=1 schedule clarification": any_schedule_clarification,
        ">=1 client clarification": any_client_clarification,
        ">=1 artifact version 2": any_artifact_v2,
    }
    print("\n  minimum passing criteria:")
    all_pass = True
    for label, ok in checks.items():
        print(f"    [{'OK' if ok else 'FAIL'}] {label}")
        all_pass = all_pass and ok
    print(f"\n  {'[ALL CRITERIA PASS]' if all_pass else '[SOME CRITERIA FAILED]'}")
    return all_pass


if __name__ == "__main__":
    all_pass = print_structural_diversity_summary()
    print()
    unittest.main(verbosity=2, exit=False)
    if not all_pass:
        sys.exit(1)
