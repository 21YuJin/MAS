"""
[Step 6.5D] Unit tests for the formal workload mock full-run: all 50 tasks
complete deterministically, strict validation passes, expected branches
match observed branches, event/graph pattern diversity meets the minimums,
active-agent coverage, difficulty/hard-budget confound reporting, adjacency
signatures, and the diagnostic-vs-LLM-input metadata boundary. No Ollama
calls anywhere in this file.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_formal_workload_mock_run.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.formal_workload_mock_run import (  # noqa: E402
    aggregate_active_agent_report, aggregate_branch_match, aggregate_difficulty_behavior,
    aggregate_event_patterns, aggregate_graph_patterns, aggregate_hard_budget_confound_report,
    aggregate_mock_execution_summary, aggregate_split_behavior, run_all_formal_mock_sessions,
    write_phase_6_5d_reports,
)
from travel_a2a.models import FORBIDDEN_METADATA_KEYS  # noqa: E402

# Runs the full 50-task mock execution ONCE for the whole module -- each
# individual task is a cheap deterministic mock (no Ollama), but running it
# 50 times per test class would be wasteful.
_OUTCOME_PAIRS = run_all_formal_mock_sessions(save_sessions=False)
_OUTCOMES = [o for o, _ in _OUTCOME_PAIRS]


class TestFullRunCompletion(unittest.TestCase):
    def test_01_all_50_tasks_loaded_and_run(self):
        self.assertEqual(len(_OUTCOMES), 50)

    def test_02_all_50_sessions_completed(self):
        completed = [o for o in _OUTCOMES if o.task_status == "completed"]
        self.assertEqual(len(completed), 50)

    def test_03_zero_strict_validation_errors(self):
        errors = [o for o in _OUTCOMES if o.strict_validation_error_count > 0]
        self.assertEqual(errors, [], [(o.task_instance_id, o.strict_validation_errors) for o in errors])

    def test_04_final_travel_plan_present_for_every_task(self):
        missing = [o.task_instance_id for o in _OUTCOMES if not o.final_travel_plan_present]
        self.assertEqual(missing, [])


class TestBranchMatch(unittest.TestCase):
    def test_05_branch_exact_match_for_every_task(self):
        report = aggregate_branch_match(_OUTCOMES)
        self.assertEqual(report["mismatch_count"], 0, report["mismatches"])


class TestPatternDiversity(unittest.TestCase):
    def test_06_at_least_10_distinct_event_patterns(self):
        report = aggregate_event_patterns(_OUTCOMES)
        self.assertGreaterEqual(report["distinct_pattern_count"], 10)
        self.assertTrue(report["meets_minimum_10"])

    def test_07_at_least_6_distinct_graph_patterns(self):
        report = aggregate_graph_patterns(_OUTCOMES)
        self.assertGreaterEqual(report["distinct_pattern_count"], 6)
        self.assertTrue(report["meets_minimum_6"])


class TestActiveAgentCoverage(unittest.TestCase):
    def test_08_coordinator_flight_hotel_always_active(self):
        report = aggregate_active_agent_report(_OUTCOMES)
        self.assertEqual(report["active_session_counts"]["travel_coordinator"], 50)
        self.assertEqual(report["active_session_counts"]["flight_agent"], 50)
        self.assertEqual(report["active_session_counts"]["hotel_agent"], 50)

    def test_09_currency_and_tours_agents_reasonably_active(self):
        report = aggregate_active_agent_report(_OUTCOMES)
        self.assertGreater(report["active_session_counts"]["currency_agent"], 15)
        self.assertGreater(report["active_session_counts"]["tours_agent"], 15)


class TestDifficultyConfoundReport(unittest.TestCase):
    def test_10_difficulty_behavior_report_generated_with_all_tiers(self):
        report = aggregate_difficulty_behavior(_OUTCOMES)
        self.assertEqual(set(report["stats_by_difficulty"].keys()), {"easy", "medium", "hard"})

    def test_11_difficulty_event_count_increases_with_tier(self):
        report = aggregate_difficulty_behavior(_OUTCOMES)
        means = {d: report["stats_by_difficulty"][d]["event_count"]["mean"] for d in ("easy", "medium", "hard")}
        self.assertLess(means["easy"], means["medium"])
        self.assertLess(means["medium"], means["hard"])

    def test_12_confound_report_never_raises_and_is_a_list(self):
        report = aggregate_difficulty_behavior(_OUTCOMES)
        self.assertIsInstance(report["confounds"], list)


class TestHardBudgetConfoundReport(unittest.TestCase):
    def test_13_hard_budget_confound_report_generated(self):
        report = aggregate_hard_budget_confound_report(_OUTCOMES)
        self.assertEqual(report["hard_task_count"], 15)
        self.assertIn("budget_revision_rate_among_hard", report)
        self.assertIn("recommend_redesign", report)


class TestAdjacencySignature(unittest.TestCase):
    def test_14_adjacency_signature_generated_and_differs_from_fixed_topology(self):
        adjacencies = {tuple(tuple(p) for p in o.adjacency_signature) for o in _OUTCOMES}
        self.assertGreater(len(adjacencies), 1, "adjacency signatures must vary across sessions")
        for o in _OUTCOMES:
            self.assertTrue(all(len(pair) == 2 for pair in o.adjacency_signature))


class TestMetadataBoundary(unittest.TestCase):
    def test_15_no_forbidden_metadata_field_in_outcome_summary(self):
        for o in _OUTCOMES[:10]:
            summary_keys = set(o.to_summary_dict().keys())
            self.assertEqual(summary_keys & FORBIDDEN_METADATA_KEYS, set())


class TestSplitBehaviorAndWriters(unittest.TestCase):
    def test_16_split_behavior_report_generated(self):
        report = aggregate_split_behavior(_OUTCOMES)
        self.assertIn("stats_by_split", report)
        self.assertIn("warnings", report)
        self.assertIsInstance(report["warnings"], list)

    def test_17_write_phase_6_5d_reports_to_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            validation_report = write_phase_6_5d_reports(_OUTCOMES, report_root=tmp)
            for fname in ("mock_execution_summary.json", "mock_execution_summary.csv",
                          "branch_match_report.json", "event_pattern_report.json",
                          "graph_pattern_report.json", "difficulty_behavior_report.json",
                          "split_behavior_report.json", "active_agent_report.json",
                          "hard_budget_confound_report.json", "phase_6_5d_validation_report.json"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, fname)), fname)
            self.assertTrue(validation_report["overall_pass"])

    def test_18_mock_execution_summary_counts_match(self):
        summary = aggregate_mock_execution_summary(_OUTCOMES)
        self.assertEqual(summary["session_count"], 50)
        self.assertEqual(summary["completed_count"], 50)
        self.assertEqual(summary["strict_validation_error_count"], 0)


class TestExistingRegression(unittest.TestCase):
    def test_19_existing_step_suites_still_importable(self):
        import test_travel_a2a_formal_workload  # noqa: F401
        import test_travel_a2a_formal_workload_validation  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
