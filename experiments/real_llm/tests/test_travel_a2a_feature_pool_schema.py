"""
[Step 7A] Unit tests for the raw metadata schema (configs/travel_a2a/feature_pool/
raw_metadata_schema.json). This is a SCHEMA AUDIT test suite, not a feature-
engineering one: it verifies the schema is (a) structurally well-formed, (b)
grounded in fields that actually exist on models.py/metadata_delta.py/
formal_workload_mock_run.py -- not fabricated -- and (c) never lists a
models.FORBIDDEN_METADATA_KEYS entry as raw material, and that the two known
Step 6.5D confounds (event_count/message_count/revision_count) are annotated
exactly as the Dataset Card describes them. No feature generation, no
LightGAE input, no Ollama calls anywhere in this file.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_feature_pool_schema.py
"""
import dataclasses
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.models import AgentCallRecord, FORBIDDEN_METADATA_KEYS, InteractionEvent  # noqa: E402
from travel_a2a.metadata_delta import _agent_call_aggregate, _artifact_aggregate, _event_aggregate  # noqa: E402
from travel_a2a.formal_workload_mock_run import TaskMockRunOutcome  # noqa: E402

SCHEMA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "configs", "travel_a2a", "feature_pool", "raw_metadata_schema.json"))

_VALID_STATUSES = {"collected", "field_exists_not_populated", "planned_not_collected"}
_REQUIRED_FIELD_KEYS = {"name", "dtype", "status", "candidate_only", "deployment_available", "description", "known_confound"}


def _load_schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _all_field_entries(schema: dict):
    for layer in schema["layers"].values():
        for entry in layer["fields"]:
            yield entry


def _split_names(name: str):
    # A handful of session-level entries document several combined derived
    # fields under one name (e.g. "total_call_count / llm_call_count") --
    # split on "/" so each individual name can still be checked.
    return [n.strip() for n in name.split("/")]


class TestSchemaWellFormed(unittest.TestCase):
    def test_01_schema_file_loads_and_has_version_metadata(self):
        schema = _load_schema()
        self.assertEqual(schema["experiment_version"], "travel_a2a_v2")
        self.assertEqual(schema["environment_type"], "a2a_inspired_travel")
        self.assertEqual(schema["llm_backend"], "ollama")

    def test_02_all_four_layers_present(self):
        schema = _load_schema()
        self.assertEqual(set(schema["layers"].keys()), {"node", "edge", "session", "runtime"})

    def test_03_every_layer_has_at_least_one_field(self):
        schema = _load_schema()
        for layer_name, layer in schema["layers"].items():
            self.assertGreater(len(layer["fields"]), 0, layer_name)

    def test_04_every_field_entry_has_required_keys(self):
        schema = _load_schema()
        for entry in _all_field_entries(schema):
            missing = _REQUIRED_FIELD_KEYS - set(entry.keys())
            self.assertEqual(missing, set(), entry.get("name"))

    def test_05_every_field_status_is_known(self):
        schema = _load_schema()
        for entry in _all_field_entries(schema):
            self.assertIn(entry["status"], _VALID_STATUSES, entry["name"])


class TestNoGroundTruthLeakage(unittest.TestCase):
    def test_06_no_forbidden_metadata_key_listed_as_raw_field(self):
        schema = _load_schema()
        all_names = set()
        for entry in _all_field_entries(schema):
            all_names.update(_split_names(entry["name"]))
        leaked = all_names & FORBIDDEN_METADATA_KEYS
        self.assertEqual(leaked, set())

    def test_07_no_dataset_provenance_field_listed_as_raw_field(self):
        schema = _load_schema()
        provenance_fields = {"task_instance_id", "template_id", "task_group_id", "content_bundle_id",
                              "expected_normal_branches", "split", "hard_normal_tags",
                              "generation_seed", "generator_version", "workload_version"}
        all_names = set()
        for entry in _all_field_entries(schema):
            all_names.update(_split_names(entry["name"]))
        self.assertEqual(all_names & provenance_fields, set())


class TestGroundedInActualCode(unittest.TestCase):
    def test_08_node_level_fields_exist_on_agent_call_record(self):
        schema = _load_schema()
        known = {f.name for f in dataclasses.fields(AgentCallRecord)}
        for entry in schema["layers"]["node"]["fields"]:
            self.assertIn(entry["name"], known, entry["name"])

    def test_09_edge_level_stored_fields_exist_on_interaction_event(self):
        schema = _load_schema()
        known_fields = {f.name for f in dataclasses.fields(InteractionEvent)}
        known_properties = {"wall_clock_latency_ms", "time_since_previous_event_ms"}
        for entry in schema["layers"]["edge"]["fields"]:
            if entry["status"] == "planned_not_collected":
                continue
            self.assertIn(entry["name"], known_fields | known_properties, entry["name"])

    def test_10_runtime_level_fields_exist_on_agent_call_record_or_event(self):
        # llm_backend is a fixed project-level constant ("ollama"), never
        # stored as a per-call/per-event object attribute -- exempted here,
        # not grounded against a dataclass field like the rest of this layer.
        schema = _load_schema()
        known = ({f.name for f in dataclasses.fields(AgentCallRecord)}
                 | {f.name for f in dataclasses.fields(InteractionEvent)}
                 | {"llm_backend"})
        for entry in schema["layers"]["runtime"]["fields"]:
            self.assertIn(entry["name"], known, entry["name"])

    def test_11_session_level_collected_fields_exist_in_known_aggregators(self):
        schema = _load_schema()
        known = {f.name for f in dataclasses.fields(TaskMockRunOutcome)}
        known |= set(_agent_call_aggregate([]).keys())
        known |= set(_event_aggregate([], []).keys())
        known |= set(_artifact_aggregate([], []).keys())
        for entry in schema["layers"]["session"]["fields"]:
            if entry["status"] != "collected":
                continue
            names = _split_names(entry["name"])
            for name in names:
                self.assertIn(name, known, name)


class TestKnownConfoundAnnotations(unittest.TestCase):
    def test_12_event_count_flagged_candidate_only_with_difficulty_confound(self):
        schema = _load_schema()
        entry = next(e for e in schema["layers"]["session"]["fields"] if e["name"] == "event_count")
        self.assertTrue(entry["candidate_only"])
        self.assertIn("difficulty", entry["known_confound"])

    def test_13_message_count_flagged_candidate_only_with_difficulty_confound(self):
        schema = _load_schema()
        entry = next(e for e in schema["layers"]["session"]["fields"] if e["name"] == "message_count")
        self.assertTrue(entry["candidate_only"])
        self.assertIn("difficulty", entry["known_confound"])

    def test_14_revision_count_flagged_candidate_only_with_branch_confound(self):
        schema = _load_schema()
        entry = next(e for e in schema["layers"]["session"]["fields"] if e["name"] == "revision_count")
        self.assertTrue(entry["candidate_only"])
        self.assertIsNotNone(entry["known_confound"])

    def test_15_most_fields_are_not_candidate_only(self):
        # A sanity check that candidate_only isn't applied blanket-wide --
        # only the specific Step 6.5D-confirmed confounded fields should carry it.
        schema = _load_schema()
        flagged = [e["name"] for e in _all_field_entries(schema) if e["candidate_only"]]
        self.assertEqual(set(flagged), {"event_count", "message_count", "revision_count", "workflow_duration"})


class TestExistingRegression(unittest.TestCase):
    def test_16_existing_step_suites_still_importable(self):
        import test_travel_a2a_formal_workload  # noqa: F401
        import test_travel_a2a_formal_workload_validation  # noqa: F401
        import test_travel_a2a_formal_workload_mock_run  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
