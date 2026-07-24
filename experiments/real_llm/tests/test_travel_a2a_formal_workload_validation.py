"""
[Step 6.5C] Unit tests for static workload validation: group-aware primary
split (with content-bundle-sharing merge), secondary unseen-template split,
shortcut-risk checks, near-duplicate cross-split detection, and the overall
static validation checklist. No Ollama/mock execution here (Phase 6.5D).

Run directly:
    python experiments/real_llm/tests/test_travel_a2a_formal_workload_validation.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from travel_a2a.formal_workload_validation import (  # noqa: E402
    UNSEEN_CONSTRAINT_COMBINATION, ShortcutIssue, _group_overlap_count, _merge_groups_sharing_content_bundle,
    build_primary_split, build_secondary_split, build_split_balance_report, load_content_bundles,
    load_task_instances, near_duplicate_report, validate_shortcut_risks, validate_workload_static, write_reports,
    write_splits,
)


class TestPrimarySplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_instances = load_task_instances()
        cls.content_bundles = load_content_bundles()
        cls.primary = build_primary_split(cls.task_instances)

    def test_01_every_task_instance_assigned_exactly_once(self):
        all_ids = (self.primary["train_task_ids"] + self.primary["validation_task_ids"]
                   + self.primary["test_task_ids"])
        self.assertEqual(len(all_ids), 50)
        self.assertEqual(len(set(all_ids)), 50)

    def test_02_no_group_overlap_across_splits(self):
        self.assertEqual(_group_overlap_count(self.primary), 0)

    def test_03_group_leakage_zero_for_task_group_id(self):
        by_id = {t.task_instance_id: t for t in self.task_instances}
        id_to_split = {}
        for s in ("train", "validation", "test"):
            for tid in self.primary[f"{s}_task_ids"]:
                id_to_split[tid] = s
        group_splits = {}
        for tid, s in id_to_split.items():
            gid = by_id[tid].task_group_id
            group_splits.setdefault(gid, set()).add(s)
        leaked = {gid: splits for gid, splits in group_splits.items() if len(splits) > 1}
        self.assertEqual(leaked, {})

    def test_04_task_group_id_is_the_split_unit_not_content_bundle_id(self):
        """[Phase 6.5D] content_bundle_id is destination-scoped and shared by
        many unrelated templates -- forcing every content-bundle-sharing
        template into one split would collapse the 35 template groups into a
        handful of giant blocks (verified during Phase 6.5D to badly damage
        difficulty/family balance). Only task_group_id is the hard
        leakage-prevention unit (test_03); content_bundle_id sharing across
        splits is expected and separately reported (test_12)."""
        self.assertEqual(self.primary["split_unit"], "task_group_id")

    def test_05_instance_counts_close_to_30_10_10(self):
        counts = self.primary["instance_counts"]
        self.assertEqual(sum(counts.values()), 50)
        # exact match happens to hold for the committed seed/generator, but
        # the contract is "as close as possible without splitting a group" --
        # allow a small tolerance so this test doesn't over-fit one seed.
        for split_name, target in self.primary["target_counts"].items():
            self.assertLessEqual(abs(counts[split_name] - target), 3)

    def test_06_split_deterministic_across_calls(self):
        p2 = build_primary_split(self.task_instances)
        self.assertEqual(self.primary["train_task_ids"], p2["train_task_ids"])
        self.assertEqual(self.primary["validation_task_ids"], p2["validation_task_ids"])
        self.assertEqual(self.primary["test_task_ids"], p2["test_task_ids"])


class TestContentBundleMerge(unittest.TestCase):
    def test_07_merge_unions_groups_sharing_a_content_bundle(self):
        task_instances = load_task_instances()
        merged = _merge_groups_sharing_content_bundle(task_instances)
        raw_group_count = len({t.task_group_id for t in task_instances})
        self.assertLessEqual(len(merged), raw_group_count)
        # every merged unit's members must all share task_group_id OR a
        # content_bundle_id transitively -- weakest checkable invariant:
        # no content_bundle_id appears in two different merged units.
        bundle_to_unit = {}
        for unit_id, members in merged.items():
            for m in members:
                if m.content_bundle_id in bundle_to_unit:
                    self.assertEqual(bundle_to_unit[m.content_bundle_id], unit_id)
                else:
                    bundle_to_unit[m.content_bundle_id] = unit_id


class TestSecondarySplit(unittest.TestCase):
    def test_08_secondary_split_holds_out_the_documented_combination(self):
        task_instances = load_task_instances()
        secondary = build_secondary_split(task_instances)
        self.assertEqual(secondary["holdout_constraint_combination"], list(UNSEEN_CONSTRAINT_COMBINATION))
        self.assertGreater(len(secondary["test_task_ids"]), 0)

    def test_09_secondary_split_does_not_holdout_a_whole_family(self):
        task_instances = load_task_instances()
        secondary = build_secondary_split(task_instances)
        self.assertTrue(secondary["all_families_still_represented_outside_holdout"])

    def test_10_secondary_split_partitions_all_50_tasks(self):
        task_instances = load_task_instances()
        secondary = build_secondary_split(task_instances)
        combined = set(secondary["test_task_ids"]) | set(secondary["train_and_validation_task_ids"])
        self.assertEqual(len(combined), 50)
        self.assertEqual(set(secondary["test_task_ids"]) & set(secondary["train_and_validation_task_ids"]), set())


class TestShortcutRisks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_instances = load_task_instances()
        cls.content_bundles = load_content_bundles()
        cls.primary = build_primary_split(cls.task_instances)
        cls.issues = validate_shortcut_risks(cls.task_instances, cls.content_bundles, cls.primary)

    def test_11_no_critical_issues(self):
        critical = [i for i in self.issues if i.severity == "critical"]
        self.assertEqual(critical, [], f"critical shortcut issues found: {[i.issue_code for i in critical]}")

    def test_12_content_bundle_reuse_leakage_reported_as_low_severity(self):
        """[Phase 6.5D] Expected to fire given destination-scoped content
        bundles shared across many templates -- accepted trade-off (see
        build_primary_split's docstring), so severity must stay 'low', never
        'critical'/'medium'."""
        matches = [i for i in self.issues if i.issue_code == "CONTENT_BUNDLE_REUSE_LEAKAGE"]
        if matches:
            self.assertEqual(matches[0].severity, "low")

    def test_13_option_position_bias_not_flagged(self):
        codes = {i.issue_code for i in self.issues}
        self.assertNotIn("OPTION_POSITION_BIAS", codes)

    def test_14_price_rank_bias_not_flagged(self):
        codes = {i.issue_code for i in self.issues}
        self.assertNotIn("PRICE_RANK_BIAS", codes)

    def test_15_difficulty_branch_confound_reported_when_present(self):
        hard = [t for t in self.task_instances if t.difficulty == "hard"]
        hard_budget = [t for t in hard if "budget_revision" in t.expected_normal_branches]
        codes = {i.issue_code for i in self.issues}
        if hard and len(hard_budget) / len(hard) >= 0.9:
            self.assertIn("DIFFICULTY_BRANCH_CONFOUND", codes)

    def test_16_all_issues_have_required_fields(self):
        for i in self.issues:
            self.assertIsInstance(i, ShortcutIssue)
            self.assertIn(i.severity, ("critical", "medium", "low", "info"))
            self.assertTrue(i.explanation)
            self.assertTrue(i.recommended_fix)


class TestNearDuplicateReport(unittest.TestCase):
    def test_17_no_cross_split_near_duplicate_violations(self):
        task_instances = load_task_instances()
        primary = build_primary_split(task_instances)
        nd = near_duplicate_report(task_instances, primary)
        self.assertEqual(nd["cross_split_violation_count"], 0, nd["cross_split_violations"])


class TestStaticValidationChecklist(unittest.TestCase):
    def test_18_overall_pass_true_for_committed_workload(self):
        task_instances = load_task_instances()
        content_bundles = load_content_bundles()
        primary = build_primary_split(task_instances)
        balance = build_split_balance_report(task_instances, primary)
        issues = validate_shortcut_risks(task_instances, content_bundles, primary)
        nd = near_duplicate_report(task_instances, primary)
        checks = validate_workload_static(task_instances, content_bundles, primary, issues, nd)
        self.assertTrue(checks["overall_pass"], checks)
        self.assertEqual(checks["task_instance_id_duplicates"], 0)
        self.assertEqual(checks["forbidden_metadata_field_leak_count"], 0)
        self.assertTrue(checks["hard_normal_coverage_in_target_range"])
        self.assertIsNotNone(balance)


class TestWriters(unittest.TestCase):
    def test_19_write_splits_and_reports_to_temp_dir(self):
        task_instances = load_task_instances()
        content_bundles = load_content_bundles()
        primary = build_primary_split(task_instances)
        secondary = build_secondary_split(task_instances)
        balance = build_split_balance_report(task_instances, primary)
        issues = validate_shortcut_risks(task_instances, content_bundles, primary)
        nd = near_duplicate_report(task_instances, primary)
        checks = validate_workload_static(task_instances, content_bundles, primary, issues, nd)

        with tempfile.TemporaryDirectory() as splits_dir, tempfile.TemporaryDirectory() as report_dir:
            write_splits(primary, secondary, balance, splits_dir=splits_dir)
            write_reports(task_instances, content_bundles, primary, balance, issues, nd, checks, report_root=report_dir)
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "primary_group_split.json")))
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "unseen_template_split.json")))
            self.assertTrue(os.path.isfile(os.path.join(report_dir, "workload_summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(report_dir, "shortcut_risk_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(report_dir, "validation_report.json")))


class TestExistingRegression(unittest.TestCase):
    def test_20_existing_step_suites_still_importable(self):
        import test_travel_a2a_formal_workload  # noqa: F401
        import test_travel_a2a_step6  # noqa: F401
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
