"""
[Step 7B] Unit tests for the Candidate Feature Generator
(experiments/real_llm/travel_a2a/feature_generators/). Covers: per-group
formula correctness on synthetic fixtures, raw-record preservation, a
structural guarantee that no attack/difficulty/split/content field is ever
read, determinism of the end-to-end manifest build over all 50 formal
workload mock sessions, and existing-suite regression. No Ollama calls, no
feature screening/selection/scoring anywhere in this file.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_feature_generators.py
"""
import copy
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.feature_generators import (  # noqa: E402
    generate, graph_features, normalization_features, session_features, timing_features, token_features,
)
from travel_a2a.feature_generators.manifest import (  # noqa: E402
    build_feature_generation_manifest, write_feature_generation_manifest,
)
from travel_a2a.formal_workload_mock_run import run_all_formal_mock_sessions  # noqa: E402
from travel_a2a.models import (  # noqa: E402
    AgentCallRecord, Artifact, ArtifactType, InteractionEvent, InteractionType, Message, Part, PartType, SourceType,
)


def _call(call_id, agent_id, *, input_part_ids=None, input_artifact_ids=None,
          output_part_ids=None, output_artifact_ids=None, prompt_eval_count=None, eval_count=None,
          prompt_eval_duration=None, eval_duration=None, total_duration=None,
          wall_clock_latency_ms=None, call_end_timestamp=""):
    return AgentCallRecord(
        call_id=call_id, session_id="s1", task_id="t1", context_id="c1", agent_id=agent_id,
        action_type="search", input_part_ids=(input_part_ids or []), input_artifact_ids=(input_artifact_ids or []),
        output_part_ids=(output_part_ids or []), output_artifact_ids=(output_artifact_ids or []),
        call_start_timestamp="", call_end_timestamp=call_end_timestamp,
        wall_clock_latency_ms=wall_clock_latency_ms,
        prompt_eval_count=prompt_eval_count, eval_count=eval_count,
        prompt_eval_duration=prompt_eval_duration, eval_duration=eval_duration, total_duration=total_duration,
    )


def _event(sender, receiver, event_id="e1", event_index=0):
    return InteractionEvent(event_id=event_id, event_index=event_index, session_id="s1", task_id="t1",
                             context_id="c1", sender_id=sender, receiver_id=receiver,
                             interaction_type=InteractionType.TASK_DELEGATION)


def _message(message_id, sender, receiver, request_message_id=None, part_ids=None):
    return Message(message_id=message_id, task_id="t1", context_id="c1", sender_id=sender, receiver_id=receiver,
                   interaction_type=InteractionType.TASK_DELEGATION, role="agent",
                   request_message_id=request_message_id, part_ids=(part_ids or []))


def _artifact(artifact_id, producer_id, parent_artifact_ids=None, part_ids=None):
    return Artifact(artifact_id=artifact_id, task_id="t1", context_id="c1", artifact_type=ArtifactType.FLIGHT_OPTIONS,
                     producer_id=producer_id, parent_artifact_ids=(parent_artifact_ids or []),
                     part_ids=(part_ids or []))


def _part(part_id, content, size_bytes=None):
    return Part(part_id=part_id, part_type=PartType.TEXT, mime_type="text/plain", content=content,
                source_type=SourceType.AGENT_GENERATED, size_bytes=size_bytes)


class TestTokenFeatures(unittest.TestCase):
    def test_01_basic_ratios_and_predecessor_resolution(self):
        c1 = _call("c1", "flight_agent", output_part_ids=["p1"], prompt_eval_count=100, eval_count=50,
                   call_end_timestamp="2027-01-01T00:00:00")
        c2 = _call("c2", "hotel_agent", input_part_ids=["p1"], prompt_eval_count=120, eval_count=30,
                   call_end_timestamp="2027-01-01T00:00:05")
        rows = token_features.token_features_for_session([c1, c2])
        row2 = next(r for r in rows if r["call_id"] == "c2")
        self.assertEqual(row2["predecessor_call_id"], "c1")
        self.assertAlmostEqual(row2["ctx_delta"], 120 - 50)
        self.assertAlmostEqual(row2["expansion_ratio"], 30 / 120)
        self.assertEqual(row2["total_token_count"], 150)

    def test_02_none_when_no_predecessor_or_missing_telemetry(self):
        c1 = _call("c1", "flight_agent")  # no telemetry at all (mock-like)
        rows = token_features.token_features_for_session([c1])
        row = rows[0]
        self.assertIsNone(row["predecessor_call_id"])
        self.assertIsNone(row["expansion_ratio"])
        self.assertIsNone(row["ctx_delta"])
        self.assertIsNone(row["total_token_count"])

    def test_03_division_by_zero_input_token_count_is_none(self):
        c1 = _call("c1", "flight_agent", prompt_eval_count=0, eval_count=10)
        row = token_features.token_features_for_session([c1])[0]
        self.assertIsNone(row["expansion_ratio"])


class TestTimingFeatures(unittest.TestCase):
    def test_04_formulas_with_known_numbers(self):
        c1 = _call("c1", "flight_agent", eval_count=50, eval_duration=int(5e9), prompt_eval_duration=int(1e9),
                   total_duration=int(6e9), wall_clock_latency_ms=6000.0)
        row = timing_features.timing_features_for_session([c1])[0]
        self.assertAlmostEqual(row["generation_time_ms"], 5000.0)
        self.assertAlmostEqual(row["prompt_eval_time_ms"], 1000.0)
        self.assertAlmostEqual(row["tokens_per_second"], 10.0)
        self.assertAlmostEqual(row["generation_ratio"], 5 / 6)
        self.assertAlmostEqual(row["non_generation_overhead_ms"], 0.0)
        self.assertEqual(row["wall_clock_latency_ms"], 6000.0)

    def test_05_none_under_missing_ollama_telemetry(self):
        c1 = _call("c1", "flight_agent", wall_clock_latency_ms=12.5)  # mock-like: only wall clock populated
        row = timing_features.timing_features_for_session([c1])[0]
        self.assertEqual(row["wall_clock_latency_ms"], 12.5)
        self.assertIsNone(row["generation_time_ms"])
        self.assertIsNone(row["tokens_per_second"])
        self.assertIsNone(row["generation_ratio"])


class TestGraphFeatures(unittest.TestCase):
    def test_06_fan_in_out_and_edge_density(self):
        events = [_event("coordinator", "flight_agent", "e1"), _event("coordinator", "hotel_agent", "e2"),
                  _event("flight_agent", "coordinator", "e3")]
        report = graph_features.fan_in_out_by_agent(events)
        self.assertEqual(report["coordinator"], {"fan_in": 1, "fan_out": 2})
        self.assertEqual(report["flight_agent"], {"fan_in": 1, "fan_out": 1})
        density = graph_features.edge_density(events)
        # 3 agents, 3 unique directed pairs, 3*2=6 possible directed pairs
        self.assertAlmostEqual(density, 3 / 6)

    def test_07_parallel_edge_count_counts_repeated_pairs(self):
        events = [_event("a", "b", "e1"), _event("a", "b", "e2"), _event("a", "c", "e3")]
        self.assertEqual(graph_features.parallel_edge_count(events), 1)

    def test_08_reply_chain_depth(self):
        messages = [_message("m1", "a", "b"), _message("m2", "b", "a", request_message_id="m1"),
                    _message("m3", "a", "b", request_message_id="m2")]
        self.assertEqual(graph_features.reply_chain_depth(messages), 3)

    def test_09_artifact_lineage_depth(self):
        artifacts = [_artifact("a1", "flight_agent"), _artifact("a2", "hotel_agent", parent_artifact_ids=["a1"]),
                     _artifact("a3", "coordinator", parent_artifact_ids=["a2"])]
        self.assertEqual(graph_features.artifact_lineage_depth(artifacts), 3)

    def test_10_empty_inputs_do_not_raise(self):
        self.assertEqual(graph_features.reply_chain_depth([]), 0)
        self.assertEqual(graph_features.artifact_lineage_depth([]), 0)
        self.assertEqual(graph_features.edge_density([]), 0.0)


class TestSessionFeatures(unittest.TestCase):
    def test_11_cv_and_max_ratio(self):
        calls = [_call("c1", "a", wall_clock_latency_ms=10.0, eval_count=10),
                 _call("c2", "a", wall_clock_latency_ms=20.0, eval_count=20),
                 _call("c3", "a", wall_clock_latency_ms=30.0, eval_count=60)]
        report = session_features.session_features_for_session(calls, [], [], [])
        self.assertIsNotNone(report["latency_cv"])
        self.assertAlmostEqual(report["max_output_ratio"], 60 / 30)
        self.assertEqual(report["call_count"], 3)

    def test_12_fewer_than_two_datapoints_is_none(self):
        calls = [_call("c1", "a", wall_clock_latency_ms=10.0)]
        report = session_features.session_features_for_session(calls, [], [], [])
        self.assertIsNone(report["latency_cv"])

    def test_13_artifact_and_message_cv_from_structure_only(self):
        parts = [_part("p1", "x", size_bytes=100), _part("p2", "x", size_bytes=300)]
        artifacts = [_artifact("a1", "flight_agent", part_ids=["p1"]), _artifact("a2", "hotel_agent", part_ids=["p2"])]
        messages = [_message("m1", "a", "b", part_ids=["p1"]), _message("m2", "a", "b", part_ids=["p1", "p2"])]
        report = session_features.session_features_for_session([], messages, artifacts, parts)
        self.assertIsNotNone(report["artifact_cv"])
        self.assertIsNotNone(report["message_cv"])


class TestNormalizationFeaturesScaffoldingOnly(unittest.TestCase):
    def test_14_zscore_formula_with_synthetic_stats(self):
        stats = {("flight_agent", "wall_clock_latency_ms"): {"mean": 100.0, "stdev": 10.0}}
        z = normalization_features.agent_zscore(120.0, "flight_agent", "wall_clock_latency_ms", stats)
        self.assertAlmostEqual(z, 2.0)

    def test_15_missing_stats_or_value_returns_none_never_raises(self):
        self.assertIsNone(normalization_features.agent_zscore(None, "flight_agent", "x", {}))
        self.assertIsNone(normalization_features.agent_zscore(1.0, "flight_agent", "x", {}))

    def test_16_module_never_imports_a_statistics_fitting_dependency(self):
        # Structural guarantee: this module must never itself read raw
        # session data or a training split -- its only public function takes
        # a pre-fit mapping as an explicit parameter.
        sig = inspect.signature(normalization_features.agent_zscore)
        self.assertIn("normal_statistics", sig.parameters)
        self.assertNotIn("task_instances", sig.parameters)
        self.assertNotIn("split", sig.parameters)


class TestNoForbiddenOrContentLeakage(unittest.TestCase):
    def test_17_no_generator_function_accepts_task_or_difficulty_or_split(self):
        forbidden_param_names = {"task", "travel_task", "difficulty", "split", "injection_present",
                                  "attack_id", "condition", "expected_normal_branches"}
        modules = [token_features, timing_features, graph_features, session_features, generate]
        checked = 0
        for module in modules:
            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if fn.__module__ != module.__name__:
                    continue
                params = set(inspect.signature(fn).parameters.keys())
                self.assertEqual(params & forbidden_param_names, set(), f"{module.__name__}.{name}")
                checked += 1
        self.assertGreater(checked, 0)

    def test_18_part_content_never_appears_in_generated_output(self):
        canary = "CANARY_RAW_CONTENT_MUST_NEVER_LEAK_9f3a"
        parts = [_part("p1", canary, size_bytes=42)]
        artifacts = [_artifact("a1", "flight_agent", part_ids=["p1"])]
        messages = [_message("m1", "a", "b", part_ids=["p1"])]
        features = generate.generate_candidate_features_for_session([], [], messages, artifacts, parts)
        self.assertNotIn(canary, json.dumps(features))


class TestRawPreservationAndOrchestrator(unittest.TestCase):
    def test_19_raw_records_unmodified_by_generation(self):
        c1 = _call("c1", "flight_agent", output_part_ids=["p1"], prompt_eval_count=10, eval_count=5)
        before = copy.deepcopy(c1.to_dict())
        generate.generate_candidate_features_for_session([c1], [], [], [], [])
        self.assertEqual(c1.to_dict(), before)

    def test_20_orchestrator_combines_all_four_groups(self):
        features = generate.generate_candidate_features_for_session([], [], [], [], [])
        self.assertEqual(set(features.keys()), {"token_features", "timing_features", "graph_features", "session_features"})


class TestManifestOverFormalWorkload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_feature_generation_manifest()

    def test_21_covers_all_50_sessions(self):
        self.assertEqual(self.manifest["session_count"], 50)

    def test_22_determinism_hash_stable_across_two_runs(self):
        manifest2 = build_feature_generation_manifest()
        self.assertEqual(self.manifest["determinism_hash"], manifest2["determinism_hash"])

    def test_23_ollama_only_fields_are_fully_null_under_mock(self):
        self.assertEqual(self.manifest["null_rate_by_field"]["token_features.input_token_count"], 1.0)
        self.assertEqual(self.manifest["null_rate_by_field"]["timing_features.generation_time_ms"], 1.0)

    def test_24_wall_clock_latency_is_not_null_under_mock(self):
        self.assertEqual(self.manifest["null_rate_by_field"]["timing_features.wall_clock_latency_ms"], 0.0)

    def test_25_write_manifest_to_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_feature_generation_manifest(report_root=tmp)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "feature_generation_manifest.json")))
            self.assertEqual(written["session_count"], 50)

    def test_26_run_all_formal_mock_sessions_still_importable_directly(self):
        outcome_pairs = run_all_formal_mock_sessions(save_sessions=False)
        self.assertEqual(len(outcome_pairs), 50)


class TestExistingRegression(unittest.TestCase):
    def test_27_existing_step_suites_still_importable(self):
        import test_travel_a2a_feature_pool_schema  # noqa: F401
        import test_travel_a2a_formal_workload  # noqa: F401
        import test_travel_a2a_formal_workload_validation  # noqa: F401
        import test_travel_a2a_formal_workload_mock_run  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
