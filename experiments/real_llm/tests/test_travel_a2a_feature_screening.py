"""
[Step 7D] Unit tests for the feature_screening package -- registry integrity,
availability classification, redundancy grouping, leakage/confound
validators, deployment feasibility, and the static screening_plan. This
suite never removes a feature, fits normal statistics, computes a real
correlation, or trains LightGAE -- Phase 7D is validation and planning only.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_feature_screening.py
"""
import copy
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.feature_generators.graph_features import edge_density  # noqa: E402
from travel_a2a.feature_generators.session_features import _cv, _max_ratio  # noqa: E402
from travel_a2a.feature_generators.timing_features import timing_features_for_session  # noqa: E402
from travel_a2a.feature_generators.token_features import token_features_for_session  # noqa: E402
from travel_a2a.feature_screening import (  # noqa: E402
    availability_validator, confound_validator, deployment_validator, leakage_validator,
    redundancy_validator, registry_validator, screening_plan,
)
from travel_a2a.feature_screening.report import (  # noqa: E402
    DEFAULT_MANIFEST_PATH, DEFAULT_RAW_SCHEMA_PATH, DEFAULT_REGISTRY_PATH, build_feature_family_summary, run_phase_7d,
)

REGISTRY_PATH = DEFAULT_REGISTRY_PATH


def _load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _minimal_entry(name, **overrides):
    entry = {
        "feature_name": name, "feature_level": "session", "feature_family": "test_family",
        "granularity": "per_session", "source_fields": ["x"], "formula": "x", "unit": "count", "dtype": "int",
        "missing_value_policy": "not_applicable", "normalization_policy": "none",
        "requires_normal_statistics": False, "deployment_available": True, "provider_specific": False,
        "content_free": True, "candidate_only": False, "known_confound": None,
        "leakage_risk": {"level": "none", "note": None}, "mock_availability": "available_in_mock",
        "ollama_required": False, "feature_role": "candidate_input", "enabled": True,
        "derived_from_same_raw_group": "test_group", "mathematically_dependent_on": [],
        "potentially_redundant_with": [],
    }
    entry.update(overrides)
    return entry


class TestRegistryIntegrityPasses(unittest.TestCase):
    def test_01_real_registry_integrity_passes(self):
        registry = _load_registry()
        report = registry_validator.validate_registry_integrity(registry)
        self.assertTrue(report["passed"], report)


class TestDuplicateDetection(unittest.TestCase):
    def test_02_duplicate_feature_name_detected(self):
        registry = {"features": [_minimal_entry("a"), _minimal_entry("a")]}
        dup = registry_validator.find_duplicate_feature_names(registry)
        self.assertEqual(dup, ["a"])


class TestDanglingDependencyDetection(unittest.TestCase):
    def test_03_dangling_dependency_detected(self):
        registry = {"features": [_minimal_entry("a", mathematically_dependent_on=["ghost"])]}
        dangling = registry_validator.find_dangling_dependencies(registry)
        self.assertEqual(dangling, [{"feature_name": "a", "missing_reference": "ghost"}])


class TestCyclicDependencyDetection(unittest.TestCase):
    def test_04_cyclic_dependency_detected(self):
        registry = {"features": [
            _minimal_entry("a", mathematically_dependent_on=["b"]),
            _minimal_entry("b", mathematically_dependent_on=["a"]),
        ]}
        cycles = registry_validator.find_cyclic_dependencies(registry)
        self.assertGreater(len(cycles), 0)

    def test_04b_acyclic_dependency_graph_reports_no_cycles(self):
        registry = {"features": [
            _minimal_entry("a", mathematically_dependent_on=["b"]),
            _minimal_entry("b", mathematically_dependent_on=[]),
        ]}
        self.assertEqual(registry_validator.find_cyclic_dependencies(registry), [])


class TestAvailabilityClassification(unittest.TestCase):
    def test_05_mock_available_feature_classified_correctly(self):
        registry = _load_registry()
        entry = next(f for f in registry["features"] if f["feature_name"] == "event_count")
        self.assertEqual(availability_validator.classify_feature_availability(entry), "available_now_mock")

    def test_06_ollama_required_feature_classified_correctly(self):
        registry = _load_registry()
        entry = next(f for f in registry["features"] if f["feature_name"] == "token_features.input_token_count")
        self.assertEqual(availability_validator.classify_feature_availability(entry), "requires_ollama_runtime")

    def test_07_normal_statistics_required_feature_classified_correctly(self):
        registry = _load_registry()
        entry = next(f for f in registry["features"] if f["feature_name"] == "normalization_features.agent_zscore")
        self.assertEqual(availability_validator.classify_feature_availability(entry), "requires_normal_statistics")

    def test_08_availability_matches_real_manifest_with_no_mismatches(self):
        registry = _load_registry()
        with open(DEFAULT_MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
        report = availability_validator.validate_feature_availability(registry, manifest)
        self.assertTrue(report["passed"], report["manifest_mismatches"])


class TestDivisionByZeroGuarded(unittest.TestCase):
    def test_09_generators_never_produce_inf_or_nan_on_zero_denominator(self):
        from travel_a2a.models import AgentCallRecord

        zero_call = AgentCallRecord(
            call_id="c1", session_id="s", task_id="t", context_id="ctx", agent_id="a", action_type="x",
            prompt_eval_count=0, eval_count=0, eval_duration=0, prompt_eval_duration=0, total_duration=0,
        )
        token_row = token_features_for_session([zero_call])[0]
        timing_row = timing_features_for_session([zero_call])[0]
        for value in list(token_row.values()) + list(timing_row.values()):
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), value)
        self.assertIsNone(token_row["expansion_ratio"])
        self.assertIsNone(timing_row["tokens_per_second"])
        self.assertIsNone(timing_row["generation_ratio"])
        # session_features._cv/_max_ratio guard against a zero mean directly
        self.assertIsNone(_cv([0.0, 0.0]))
        self.assertIsNone(_max_ratio([0.0, 0.0]))
        self.assertEqual(edge_density([]), 0.0)


class TestRedundancyGrouping(unittest.TestCase):
    def test_10_redundancy_groups_generated_from_real_registry(self):
        registry = _load_registry()
        groups = redundancy_validator.build_redundancy_groups(registry)
        self.assertGreater(len(groups), 0)
        member_pairs = [set(g["members"]) for g in groups]
        self.assertTrue(any({"repeated_pair_count", "graph_features.parallel_edge_count"} <= m for m in member_pairs))
        self.assertTrue(any({"total_call_count", "session_features.call_count"} <= m for m in member_pairs))

    def test_11_every_group_has_auto_remove_false(self):
        registry = _load_registry()
        groups = redundancy_validator.build_redundancy_groups(registry)
        self.assertTrue(all(g["auto_remove"] is False for g in groups))
        # and the registry itself is untouched
        registry_after = _load_registry()
        self.assertEqual(len(registry["features"]), len(registry_after["features"]))


class TestLeakageValidator(unittest.TestCase):
    def test_12_real_registry_has_no_leakage(self):
        registry = _load_registry()
        report = leakage_validator.validate_no_leakage(registry)
        self.assertTrue(report["passed"], report["findings"])

    def test_13_source_field_referencing_difficulty_or_split_is_rejected(self):
        poisoned = _minimal_entry("poisoned", source_fields=["TaskInstance.difficulty"])
        hits = leakage_validator.scan_feature_for_leakage(poisoned)
        self.assertIn("difficulty", hits)

        poisoned2 = _minimal_entry("poisoned2", formula="value computed using the split assignment")
        hits2 = leakage_validator.scan_feature_for_leakage(poisoned2)
        self.assertIn("split", hits2)

    def test_13b_known_confound_annotation_is_not_itself_flagged_as_leakage(self):
        # known_confound is intentionally out of scan scope -- it NAMES a
        # confound, it does not use it as computation input.
        entry = _minimal_entry("event_count_like", known_confound=["difficulty"])
        self.assertEqual(leakage_validator.scan_feature_for_leakage(entry), [])


class TestConfoundReport(unittest.TestCase):
    def test_14_confirmed_confound_report_generated(self):
        registry = _load_registry()
        rows = confound_validator.confirmed_confound_report(registry)
        names = {r["feature_name"] for r in rows}
        self.assertEqual(names, {"event_count", "message_count", "revision_count"})
        for r in rows:
            self.assertEqual(r["severity"], "confirmed")

    def test_14b_additional_watch_is_warning_only_and_does_not_mutate_registry(self):
        registry = _load_registry()
        rows = confound_validator.additional_watch_report(registry)
        self.assertTrue(all(r["severity"] == "warning_only" for r in rows))
        registry_after = _load_registry()
        self.assertEqual(registry, registry_after)


class TestDeploymentFeasibility(unittest.TestCase):
    def test_15_classification_matches_expected_categories(self):
        registry = _load_registry()
        by_name = {f["feature_name"]: f for f in registry["features"]}
        self.assertEqual(deployment_validator.classify_deployment_feasibility(by_name["model_name"]), "diagnostic_only")
        self.assertEqual(
            deployment_validator.classify_deployment_feasibility(by_name["token_features.input_token_count"]),
            "ollama_specific")
        self.assertEqual(
            deployment_validator.classify_deployment_feasibility(by_name["graph_features.edge_density"]), "portable")
        self.assertEqual(
            deployment_validator.classify_deployment_feasibility(by_name["normalization_features.agent_zscore"]),
            "offline_only")

    def test_16_collection_context_features_excluded_from_model_input(self):
        registry = _load_registry()
        for entry in registry["features"]:
            if entry["feature_level"] == "runtime":
                self.assertEqual(deployment_validator.classify_deployment_feasibility(entry), "diagnostic_only")
                self.assertFalse(entry["enabled"])
                self.assertEqual(entry["feature_role"], "collection_context")


class TestScreeningPlan(unittest.TestCase):
    def test_17_screening_plan_is_well_formed(self):
        plan = screening_plan.build_screening_plan()
        self.assertEqual(plan["status"], "PLAN_ONLY_NOT_EXECUTED")
        self.assertEqual(len(plan["stages"]), 10)
        stage_numbers = [s["screening_stage"] for s in plan["stages"]]
        self.assertEqual(stage_numbers, list(range(1, 11)))
        required = {"screening_stage", "criterion", "input_scope", "data_source", "threshold", "action",
                    "manual_review_required"}
        for stage in plan["stages"]:
            self.assertEqual(required - set(stage.keys()), set())

    def test_18_attack_data_never_used_as_a_screening_source(self):
        plan = screening_plan.build_screening_plan()
        self.assertIn("attack_data_usage_rule", plan["rules"])
        for stage in plan["stages"]:
            self.assertNotIn("attack session", stage["data_source"].lower())
            self.assertNotIn("attack result", stage["data_source"].lower())
        self.assertIn("never", plan["rules"]["attack_data_usage_rule"].lower())


class TestFeatureFamilySummary(unittest.TestCase):
    def test_19_family_summary_covers_all_52_features(self):
        registry = _load_registry()
        summary = build_feature_family_summary(registry)
        expected_families = {"raw_session_aggregate", "agent_call_aggregate", "token_features", "timing_features",
                              "graph_features", "session_dispersion", "normalization_features", "runtime_context"}
        self.assertEqual(set(summary.keys()), expected_families)
        self.assertEqual(sum(v["total_count"] for v in summary.values()), len(registry["features"]))


class TestRunPhase7DEndToEnd(unittest.TestCase):
    def test_20_run_phase_7d_writes_all_reports_to_temp_dirs(self):
        with tempfile.TemporaryDirectory() as report_root, tempfile.TemporaryDirectory() as config_root:
            result = run_phase_7d(registry_path=DEFAULT_REGISTRY_PATH, raw_schema_path=DEFAULT_RAW_SCHEMA_PATH,
                                   manifest_path=DEFAULT_MANIFEST_PATH, report_root=report_root, config_root=config_root)
            for fname in ("registry_validation_report.json", "feature_availability_report.csv",
                          "redundancy_groups.json", "leakage_validation_report.json", "confound_risk_report.json",
                          "deployment_feasibility_report.csv", "feature_family_summary.json", "phase_7d_summary.md"):
                self.assertTrue(os.path.isfile(os.path.join(report_root, fname)), fname)
            self.assertTrue(os.path.isfile(os.path.join(config_root, "screening_plan.json")))
            self.assertTrue(result["registry_integrity"]["passed"])
            self.assertTrue(result["leakage"]["passed"])

    def test_20b_run_phase_7d_never_mutates_the_registry_file_on_disk(self):
        before = _load_registry()
        with tempfile.TemporaryDirectory() as report_root, tempfile.TemporaryDirectory() as config_root:
            run_phase_7d(report_root=report_root, config_root=config_root)
        after = _load_registry()
        self.assertEqual(before, after)


class TestExistingRegression(unittest.TestCase):
    def test_21_existing_step_suites_still_importable(self):
        import test_travel_a2a_feature_generators  # noqa: F401
        import test_travel_a2a_feature_pool_schema  # noqa: F401
        import test_travel_a2a_feature_registry  # noqa: F401
        import test_travel_a2a_formal_workload_mock_run  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
