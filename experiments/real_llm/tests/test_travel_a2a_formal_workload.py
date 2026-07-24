"""
[Step 6.5B] Unit tests for FormalWorkloadGenerator -- deterministic
generation of the 50 formal TaskInstance objects and their content bundles.
No mock/Ollama execution here (that's Phase 6.5D); these tests check the
STATIC generation-step guarantees from Step 6.5B section 16: exact
distribution match, ID uniqueness, date/currency consistency, minimum option
counts, hard-normal coverage, the diagnostic-vs-LLM-input field boundary, and
determinism.

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_formal_workload.py
"""
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.formal_workload_generator import (  # noqa: E402
    _KRW_RATE, _LODGING_BUDGET_FRACTION, generate_formal_workload, load_spec, materialize,
)
from travel_a2a.formal_workload_models import TaskInstance, TaskTemplate  # noqa: E402
from travel_a2a.models import FORBIDDEN_METADATA_KEYS  # noqa: E402


def _bundle_tag(bundle_id: str) -> str:
    return hashlib.sha256(bundle_id.encode("utf-8")).hexdigest()[:4].upper()


class TestSpecLoading(unittest.TestCase):
    def test_01_load_spec_loads_all_ten_files(self):
        spec = load_spec()
        for key in ("task_family_spec", "difficulty_criteria", "destination_catalog",
                    "branch_distribution_target", "content_bundle_spec", "split_policy",
                    "hard_normal_tag_taxonomy", "attack_applicability_plan", "dataset_policy",
                    "formal_collection_plan"):
            self.assertIn(key, spec)
        self.assertEqual(spec["task_family_spec"]["formal_task_count"], 50)


class TestDeterminism(unittest.TestCase):
    def test_02_same_seed_same_determinism_hash(self):
        r1 = generate_formal_workload(seed=42)
        r2 = generate_formal_workload(seed=42)
        self.assertEqual(r1.workload_manifest["determinism_hash"], r2.workload_manifest["determinism_hash"])

    def test_03_different_seed_different_determinism_hash(self):
        r1 = generate_formal_workload(seed=42)
        r2 = generate_formal_workload(seed=99)
        self.assertNotEqual(r1.workload_manifest["determinism_hash"], r2.workload_manifest["determinism_hash"])


class TestDistributionExactMatch(unittest.TestCase):
    """[Step 6.5B-3] family/difficulty must match the Phase 6.5A spec
    EXACTLY -- not approximately."""

    @classmethod
    def setUpClass(cls):
        cls.result = generate_formal_workload(seed=42)
        cls.spec = load_spec()

    def test_04_exactly_50_task_instances(self):
        self.assertEqual(len(self.result.task_instances), 50)

    def test_05_family_distribution_exact(self):
        expected = {f["template_family"]: f["target_task_count"] for f in self.spec["task_family_spec"]["families"]}
        self.assertEqual(self.result.workload_manifest["family_distribution"], expected)

    def test_06_difficulty_distribution_exact(self):
        self.assertEqual(self.result.workload_manifest["difficulty_distribution"],
                          self.spec["task_family_spec"]["difficulty_totals"])

    def test_07_destination_count_at_least_12(self):
        self.assertGreaterEqual(self.result.workload_manifest["destination_count"], 12)


class TestIdentityAndUniqueness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = generate_formal_workload(seed=42)

    def test_08_task_instance_id_no_duplicates(self):
        ids = [t.task_instance_id for t in self.result.task_instances]
        self.assertEqual(len(ids), len(set(ids)))

    def test_09_content_option_id_no_duplicates_across_bundles(self):
        opt_ids = ([o["option_id"] for o in self.result.content_bundles["flights"]]
                   + [o["option_id"] for o in self.result.content_bundles["hotels"]]
                   + [o["option_id"] for o in self.result.content_bundles["tours"]])
        self.assertEqual(len(opt_ids), len(set(opt_ids)))

    def test_10_task_group_id_present_on_every_instance(self):
        self.assertTrue(all(t.task_group_id for t in self.result.task_instances))

    def test_10b_task_group_id_groups_by_template_not_destination(self):
        """[Step 6.5-10] Two instances sharing template_family/required_services/
        constraint_types but differing only in destination/duration/budget must
        land in the SAME task_group_id -- grouping by the finer per-bundle key
        would make every group a singleton and defeat group-aware splitting."""
        groups = {}
        for t in self.result.task_instances:
            groups.setdefault(t.task_group_id, []).append(t)
        self.assertLess(len(groups), len(self.result.task_instances),
                         "task_group_id must not be 1:1 with task_instance_id")
        multi_member_groups = [members for members in groups.values() if len(members) > 1]
        self.assertGreater(len(multi_member_groups), 0)
        for members in multi_member_groups:
            destinations = {m.destination for m in members}
            self.assertGreater(len(destinations), 1,
                                "a multi-member task_group should typically span >1 destination")


class TestContentConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = generate_formal_workload(seed=42)

    def test_11_required_service_content_exists_per_destination(self):
        service_to_bundle = {"flight": "flights", "hotel": "hotels", "tours": "tours"}
        for t in self.result.task_instances:
            for svc in t.required_services:
                bundle_key = service_to_bundle.get(svc)
                if bundle_key is None:
                    continue
                self.assertTrue(
                    any(o["destination"] == t.destination for o in self.result.content_bundles[bundle_key]),
                    f"{t.task_instance_id}: no {bundle_key} content for {t.destination}")

    def test_12_minimum_option_counts_per_bundle(self):
        from collections import defaultdict
        counts = defaultdict(int)
        for kind, key in (("flight", "flights"), ("hotel", "hotels"), ("tour", "tours")):
            for o in self.result.content_bundles[key]:
                tag = o["option_id"].split("_")[2]
                counts[(kind, o["destination"], tag)] += 1
        minimums = {"flight": 3, "hotel": 3, "tour": 3}
        for (kind, dest, tag), n in counts.items():
            self.assertGreaterEqual(n, minimums[kind], f"{kind} bundle for {dest}/{tag} has only {n} options")

    def test_13_date_consistency_departure_before_return(self):
        for t in self.result.task_instances:
            dep = dt.date.fromisoformat(t.departure_date)
            ret = dt.date.fromisoformat(t.return_date)
            self.assertGreater(ret, dep, f"{t.task_instance_id}: return_date not after departure_date")

    def test_14_currency_pair_exists_for_every_task(self):
        pairs = {c["pair"] for c in self.result.content_bundles["currency"]}
        for t in self.result.task_instances:
            self.assertIn(f"{t.budget_currency}/{t.target_currency}", pairs)

    def test_15_synthetic_provenance_on_every_content_record(self):
        for records in self.result.content_bundles.values():
            for r in records:
                self.assertEqual(r.get("source_id"), "generated_fixture")


class TestBranchTriggerArithmetic(unittest.TestCase):
    """[Step 6.5-10] budget/integration conflict triggers are exact-arithmetic
    guaranteed by construction -- verified here structurally (real
    workflow_policy.py execution is Phase 6.5D's job, not this one's)."""

    @classmethod
    def setUpClass(cls):
        cls.result = generate_formal_workload(seed=42)

    def test_16_budget_revision_tasks_actually_exceed_lodging_budget(self):
        checked = 0
        for t in self.result.task_instances:
            if "budget_revision" not in t.expected_normal_branches:
                continue
            tag = _bundle_tag(t.content_bundle_id)
            hotels = [o for o in self.result.content_bundles["hotels"]
                      if o["destination"] == t.destination and f"_{tag}_" in o["option_id"]]
            cheapest = min(h["total_price"] for h in hotels)
            lodging_budget = round(t.budget_amount * _KRW_RATE[t.target_currency] * _LODGING_BUDGET_FRACTION, 2)
            self.assertGreater(cheapest, lodging_budget, f"{t.task_instance_id}: budget_revision not actually triggered")
            checked += 1
        self.assertGreater(checked, 0)

    def test_17_non_budget_revision_tasks_fit_within_lodging_budget(self):
        checked = 0
        for t in self.result.task_instances:
            if "budget_revision" in t.expected_normal_branches:
                continue
            tag = _bundle_tag(t.content_bundle_id)
            hotels = [o for o in self.result.content_bundles["hotels"]
                      if o["destination"] == t.destination and f"_{tag}_" in o["option_id"]]
            cheapest = min(h["total_price"] for h in hotels)
            lodging_budget = round(t.budget_amount * _KRW_RATE[t.target_currency] * _LODGING_BUDGET_FRACTION, 2)
            self.assertLessEqual(cheapest, lodging_budget, f"{t.task_instance_id}: unexpectedly exceeds lodging budget")
            checked += 1
        self.assertGreater(checked, 0)

    def test_18_integration_revision_tasks_have_zero_in_window_tours(self):
        checked = 0
        for t in self.result.task_instances:
            if "integration_revision" not in t.expected_normal_branches:
                continue
            tag = _bundle_tag(t.content_bundle_id)
            tours = [o for o in self.result.content_bundles["tours"]
                     if o["destination"] == t.destination and f"_{tag}_" in o["option_id"]]
            in_window = [o for o in tours if t.departure_date <= o["date"] <= t.return_date]
            self.assertEqual(len(in_window), 0, f"{t.task_instance_id}: expected zero in-window tours")
            checked += 1
        self.assertGreater(checked, 0)


class TestHardNormalCoverage(unittest.TestCase):
    def test_19_hard_normal_coverage_within_target_and_not_confined_to_one_family(self):
        result = generate_formal_workload(seed=42)
        tagged = [t for t in result.task_instances if t.hard_normal_tags]
        self.assertGreaterEqual(len(tagged), 10)
        self.assertLessEqual(len(tagged), 15)
        families_tagged = {t.task_category for t in tagged}
        self.assertGreater(len(families_tagged), 1, "hard_normal_tags must not be confined to one task_category")


class TestDiagnosticLLMBoundary(unittest.TestCase):
    def test_20_llm_input_view_excludes_forbidden_and_diagnostic_fields(self):
        result = generate_formal_workload(seed=42)
        diagnostic_only = {"task_instance_id", "template_id", "task_group_id", "content_bundle_id",
                            "expected_normal_branches", "split", "hard_normal_tags",
                            "generation_seed", "generator_version", "schema_version"}
        for t in result.task_instances[:5]:
            llm_keys = set(t.to_travel_request_kwargs().keys())
            self.assertEqual(llm_keys & FORBIDDEN_METADATA_KEYS, set())
            self.assertEqual(llm_keys & diagnostic_only, set())


class TestRoundTripAndMaterialization(unittest.TestCase):
    def test_21_task_instance_json_round_trip(self):
        result = generate_formal_workload(seed=42)
        for t in result.task_instances[:5]:
            restored = TaskInstance.from_dict(json.loads(json.dumps(t.to_dict())))
            self.assertEqual(restored, t)

    def test_22_task_template_json_round_trip(self):
        result = generate_formal_workload(seed=42)
        for tmpl in result.task_templates[:5]:
            restored = TaskTemplate.from_dict(json.loads(json.dumps(tmpl.to_dict())))
            self.assertEqual(restored, tmpl)

    def test_23_workload_manifest_has_required_keys(self):
        result = generate_formal_workload(seed=42)
        for key in ("workload_version", "schema_version", "generator_version", "generation_seed",
                    "task_count", "template_count", "task_group_count", "destination_count",
                    "family_distribution", "difficulty_distribution", "branch_distribution",
                    "hard_normal_coverage", "service_combination_distribution", "content_bundle_count",
                    "spec_hashes", "determinism_hash"):
            self.assertIn(key, result.workload_manifest)

    def test_24_materialize_writes_all_expected_files(self):
        result = generate_formal_workload(seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            materialize(result, output_dir=tmp, git_commit="test_commit")
            self.assertTrue(os.path.isfile(os.path.join(tmp, "task_templates", "task_templates.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "task_instances", "task_instances.json")))
            for name in ("flights", "hotels", "tours", "currency", "policies"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "content", f"{name}.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "manifests", "workload_manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "manifests", "generation_report.json")))
            with open(os.path.join(tmp, "manifests", "workload_manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["git_commit"], "test_commit")


class TestExistingRegression(unittest.TestCase):
    def test_25_existing_step_suites_still_importable(self):
        import test_travel_a2a_attacks  # noqa: F401
        import test_travel_a2a_step6  # noqa: F401
        import test_travel_a2a_workflow  # noqa: F401
        self.assertTrue(hasattr(test_travel_a2a_workflow, "FIXTURE_INDEX"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
