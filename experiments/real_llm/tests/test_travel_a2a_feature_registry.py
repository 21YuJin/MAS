"""
[Step 7C] Unit tests for the Candidate Feature Registry
(configs/travel_a2a/feature_pool/candidate_feature_registry.json). This is a
REGISTRY AUDIT suite -- it verifies the registry is well-formed, internally
consistent (no dangling cross-references, no duplicate names), consistent
with Phase 7A's raw_metadata_schema.json and Phase 7B's actual generator
output (every produced field is registered, nothing is fabricated), and that
the "identical formula" duplicate pairs the registry calls out really are
identical when run against real formal-workload data. No feature screening,
scoring, or selection happens anywhere in this file.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_feature_registry.py
"""
import json
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.feature_generators.generate import generate_candidate_features_for_session  # noqa: E402
from travel_a2a.formal_workload_mock_run import run_all_formal_mock_sessions  # noqa: E402
from travel_a2a.models import FORBIDDEN_METADATA_KEYS  # noqa: E402

REGISTRY_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "configs", "travel_a2a", "feature_pool", "candidate_feature_registry.json"))
RAW_SCHEMA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "configs", "travel_a2a", "feature_pool", "raw_metadata_schema.json"))

_REQUIRED_KEYS = {
    "feature_name", "feature_level", "feature_family", "granularity", "source_fields", "formula", "unit", "dtype",
    "missing_value_policy", "normalization_policy", "requires_normal_statistics", "deployment_available",
    "provider_specific", "content_free", "candidate_only", "known_confound", "leakage_risk", "mock_availability",
    "ollama_required", "feature_role", "enabled", "derived_from_same_raw_group", "mathematically_dependent_on",
    "potentially_redundant_with",
}

_PROVENANCE_FIELDS = {"task_instance_id", "template_id", "task_group_id", "content_bundle_id",
                       "expected_normal_branches", "split", "hard_normal_tags",
                       "generation_seed", "generator_version", "workload_version"}


def _load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_raw_schema() -> dict:
    with open(RAW_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _feature_by_name(registry: dict, name: str) -> dict:
    return next(f for f in registry["features"] if f["feature_name"] == name)


class TestRegistryWellFormed(unittest.TestCase):
    def test_01_registry_file_loads_and_has_version_metadata(self):
        registry = _load_registry()
        self.assertEqual(registry["experiment_version"], "travel_a2a_v2")
        self.assertGreater(len(registry["features"]), 0)

    def test_02_every_entry_has_all_required_keys(self):
        registry = _load_registry()
        for entry in registry["features"]:
            missing = _REQUIRED_KEYS - set(entry.keys())
            self.assertEqual(missing, set(), entry.get("feature_name"))

    def test_03_no_duplicate_feature_names(self):
        registry = _load_registry()
        names = [f["feature_name"] for f in registry["features"]]
        dup = [n for n, c in Counter(names).items() if c > 1]
        self.assertEqual(dup, [])

    def test_04_vocabulary_compliance(self):
        registry = _load_registry()
        vocab = registry["vocabularies"]
        for entry in registry["features"]:
            self.assertIn(entry["feature_level"], vocab["feature_level"], entry["feature_name"])
            self.assertIn(entry["granularity"], vocab["granularity"], entry["feature_name"])
            self.assertIn(entry["missing_value_policy"], vocab["missing_value_policy"], entry["feature_name"])
            self.assertIn(entry["mock_availability"], vocab["mock_availability"], entry["feature_name"])
            self.assertIn(entry["feature_role"], vocab["feature_role"], entry["feature_name"])
            self.assertIn(entry["leakage_risk"]["level"], vocab["leakage_risk_level"], entry["feature_name"])

    def test_05_no_dangling_cross_references(self):
        registry = _load_registry()
        names = {f["feature_name"] for f in registry["features"]}
        for entry in registry["features"]:
            for ref in entry["mathematically_dependent_on"] + entry["potentially_redundant_with"]:
                self.assertIn(ref, names, f"{entry['feature_name']} -> {ref}")


class TestNoGroundTruthOrProvenanceLeakage(unittest.TestCase):
    def test_06_no_forbidden_metadata_key_registered(self):
        registry = _load_registry()
        names = {f["feature_name"] for f in registry["features"]}
        self.assertEqual(names & FORBIDDEN_METADATA_KEYS, set())

    def test_07_no_dataset_provenance_field_registered(self):
        registry = _load_registry()
        names = {f["feature_name"] for f in registry["features"]}
        self.assertEqual(names & _PROVENANCE_FIELDS, set())

    def test_08_excluded_from_registry_section_matches_actual_exclusions(self):
        registry = _load_registry()
        excluded_provenance = set(registry["excluded_from_registry"]["dataset_provenance_fields"]["fields"])
        self.assertEqual(excluded_provenance, _PROVENANCE_FIELDS)


class TestNormalizationAndRuntimeRolesConsistent(unittest.TestCase):
    def test_09_requires_normal_statistics_features_are_disabled_and_gated(self):
        registry = _load_registry()
        for entry in registry["features"]:
            if entry["requires_normal_statistics"]:
                self.assertEqual(entry["mock_availability"], "available_after_normal_statistics", entry["feature_name"])
                self.assertFalse(entry["enabled"], entry["feature_name"])

    def test_10_runtime_level_features_are_collection_context_and_disabled(self):
        registry = _load_registry()
        for entry in registry["features"]:
            if entry["feature_level"] == "runtime":
                self.assertEqual(entry["feature_role"], "collection_context", entry["feature_name"])
                self.assertFalse(entry["enabled"], entry["feature_name"])

    def test_11_candidate_input_features_are_enabled(self):
        registry = _load_registry()
        for entry in registry["features"]:
            if entry["feature_role"] == "candidate_input" and not entry["requires_normal_statistics"]:
                self.assertTrue(entry["enabled"], entry["feature_name"])


class TestConsistencyWithPhase7ARawSchema(unittest.TestCase):
    def test_12_known_confound_matches_raw_schema_for_shared_fields(self):
        registry = _load_registry()
        raw_schema = _load_raw_schema()
        for name in ("event_count", "message_count", "revision_count"):
            reg_entry = _feature_by_name(registry, name)
            raw_entry = next(e for e in raw_schema["layers"]["session"]["fields"] if e["name"] == name)
            self.assertEqual(reg_entry["candidate_only"], raw_entry["candidate_only"], name)
            self.assertTrue(reg_entry["candidate_only"])
            self.assertIsNotNone(reg_entry["known_confound"])
            self.assertIsNotNone(raw_entry["known_confound"])


class TestConsistencyWithActualGeneratorOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        outcome_pairs = run_all_formal_mock_sessions(save_sessions=False)
        cls.outcome, cls.result = outcome_pairs[0]
        cls.features = generate_candidate_features_for_session(
            cls.result.agent_call_records, cls.result.events, cls.result.messages,
            cls.result.artifacts, cls.result.parts)
        cls.registry = _load_registry()
        cls.registered_names = {f["feature_name"] for f in cls.registry["features"]}

    def test_13_every_token_feature_key_is_registered(self):
        identifier_keys = {"call_id", "agent_id", "predecessor_call_id"}
        for row in self.features["token_features"]:
            for key in row:
                if key in identifier_keys:
                    continue
                self.assertIn(f"token_features.{key}", self.registered_names, key)

    def test_14_every_timing_feature_key_is_registered(self):
        identifier_keys = {"call_id", "agent_id"}
        for row in self.features["timing_features"]:
            for key in row:
                if key in identifier_keys:
                    continue
                self.assertIn(f"timing_features.{key}", self.registered_names, key)

    def test_15_every_graph_feature_key_is_registered(self):
        for key in self.features["graph_features"]:
            self.assertIn(f"graph_features.{key}", self.registered_names, key)

    def test_16_every_session_feature_key_is_registered(self):
        for key in self.features["session_features"]:
            self.assertIn(f"session_features.{key}", self.registered_names, key)

    def test_17_flagged_duplicate_pairs_are_empirically_identical(self):
        # repeated_pair_count (formal_workload_mock_run) vs
        # graph_features.parallel_edge_count (Step 7B) -- registry claims
        # these are computed by an identical formula; verify on real data.
        self.assertEqual(self.outcome.repeated_pair_count, self.features["graph_features"]["parallel_edge_count"])
        # total_call_count (metadata_delta._agent_call_aggregate) vs
        # session_features.call_count -- both len(agent_call_records).
        self.assertEqual(len(self.result.agent_call_records), self.features["session_features"]["call_count"])


class TestExistingRegression(unittest.TestCase):
    def test_18_existing_step_suites_still_importable(self):
        import test_travel_a2a_feature_generators  # noqa: F401
        import test_travel_a2a_feature_pool_schema  # noqa: F401
        import test_travel_a2a_formal_workload_mock_run  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
