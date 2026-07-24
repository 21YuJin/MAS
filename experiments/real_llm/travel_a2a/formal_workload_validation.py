"""
[Step 6.5C] Static validation for the formal workload -- no Ollama calls, no
mock execution (that's Phase 6.5D). Covers:

  1. primary_group_split -- group-aware (task_group_id, never task_instance_id
     or session), stratified best-effort, targeting 30/10/10 INSTANCE counts.
     Group sizes (1/2/3) may make exact 30/10/10 unreachable -- when so, the
     deviation and cause are reported, never silently forced by splitting a
     group.
  2. unseen_template_split -- a constraint-combination holdout (NOT a whole
     template_family holdout, per Step 6.5-12's explicit warning), reported
     separately and never summed with primary_group_split results.
  3. validate_shortcut_risks() -- the 8 checks from Step 6.5-16.
  4. near_duplicate_report() -- structural-signature near-duplicates that
     land in different primary splits.
  5. validate_workload_static() -- the Step 6.5-23 static checklist.
"""
import dataclasses
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .formal_workload_generator import OUTPUT_DIR_DEFAULT, SPEC_DIR_DEFAULT, load_spec
from .formal_workload_models import TaskInstance
from .models import FORBIDDEN_METADATA_KEYS

SPLITS_DIR_DEFAULT = os.path.join(OUTPUT_DIR_DEFAULT, "splits")
REPORT_ROOT_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reports", "travel_a2a", "formal_workload"))

PRIMARY_SPLIT_TARGETS = {"train": 30, "validation": 10, "test": 10}


def load_task_instances(output_dir: str = OUTPUT_DIR_DEFAULT) -> List[TaskInstance]:
    with open(os.path.join(output_dir, "task_instances", "task_instances.json"), encoding="utf-8") as f:
        return [TaskInstance.from_dict(d) for d in json.load(f)]


def load_content_bundles(output_dir: str = OUTPUT_DIR_DEFAULT) -> Dict[str, Any]:
    bundles = {}
    for name in ("flights", "hotels", "tours", "currency", "policies"):
        with open(os.path.join(output_dir, "content", f"{name}.json"), encoding="utf-8") as f:
            bundles[name] = json.load(f)
    return bundles


# ══════════════════════════════════════════════════════════════════════════
# 1. Primary group-aware split
# ══════════════════════════════════════════════════════════════════════════


def _merge_groups_sharing_content_bundle(task_instances: List[TaskInstance]) -> Dict[str, List[TaskInstance]]:
    """[CONTENT_BUNDLE_REUSE_LEAKAGE prevention] task_group_id groups by
    TEMPLATE (Phase 6.5B), but two different templates can independently land
    on the same (destination, duration_bucket, content_profile) and share a
    content_bundle_id purely by chance. If those two task_groups were then
    assigned to different splits, raw flight/hotel/tour content would repeat
    verbatim across train and test. Union-find merges every task_group that
    shares >=1 content_bundle_id into one split unit BEFORE bin-packing, so
    this can never happen -- the split unit key is the union-find root's
    task_group_id (arbitrary but deterministic, since union-by-sorted-id)."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic: lexicographically smaller id becomes root
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    group_ids = sorted({ti.task_group_id for ti in task_instances})
    for gid in group_ids:
        parent.setdefault(gid, gid)

    bundle_to_groups: Dict[str, set] = defaultdict(set)
    for ti in task_instances:
        bundle_to_groups[ti.content_bundle_id].add(ti.task_group_id)
    for gids in bundle_to_groups.values():
        gids = sorted(gids)
        for other in gids[1:]:
            union(gids[0], other)

    merged: Dict[str, List[TaskInstance]] = defaultdict(list)
    for ti in task_instances:
        merged[find(ti.task_group_id)].append(ti)
    return merged


def build_primary_split(task_instances: List[TaskInstance], split_seed: int = 7) -> Dict[str, Any]:
    """Greedy largest-deficit bin-packing over CONTENT-BUNDLE-MERGED split
    units (see _merge_groups_sharing_content_bundle) -- deterministic (fixed
    iteration order: units sorted by (-size, unit_id), so split_seed does not
    currently affect the outcome but is recorded for provenance and future
    tie-breaking rule changes)."""
    groups = _merge_groups_sharing_content_bundle(task_instances)

    ordered_group_ids = sorted(groups.keys(), key=lambda gid: (-len(groups[gid]), gid))
    counts = {"train": 0, "validation": 0, "test": 0}
    assignment: Dict[str, str] = {}  # split unit id -> split

    for gid in ordered_group_ids:
        size = len(groups[gid])
        # deficit = how far below target (in remaining "room"); pick the
        # split with the largest deficit-to-target ratio, tie-broken by
        # fixed order train->validation->test then by lowest current count.
        deficits = {s: PRIMARY_SPLIT_TARGETS[s] - counts[s] for s in ("train", "validation", "test")}
        best_split = max(("train", "validation", "test"), key=lambda s: (deficits[s], -counts[s]))
        assignment[gid] = best_split
        counts[best_split] += size

    train_ids = sorted(ti.task_instance_id for gid in ordered_group_ids if assignment[gid] == "train" for ti in groups[gid])
    val_ids = sorted(ti.task_instance_id for gid in ordered_group_ids if assignment[gid] == "validation" for ti in groups[gid])
    test_ids = sorted(ti.task_instance_id for gid in ordered_group_ids if assignment[gid] == "test" for ti in groups[gid])

    return {
        "split_seed": split_seed,
        "split_unit": "task_group_id, merged across groups sharing a content_bundle_id",
        "split_unit_count": len(ordered_group_ids),
        "split_unit_assignment": assignment,
        "train_task_ids": train_ids,
        "validation_task_ids": val_ids,
        "test_task_ids": test_ids,
        "instance_counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "target_counts": dict(PRIMARY_SPLIT_TARGETS),
    }


def build_split_balance_report(task_instances: List[TaskInstance], primary_split: dict) -> dict:
    by_id = {t.task_instance_id: t for t in task_instances}
    id_to_split = {}
    for split_name in ("train", "validation", "test"):
        for tid in primary_split[f"{split_name}_task_ids"]:
            id_to_split[tid] = split_name

    def _dist(key_fn):
        out = {"train": defaultdict(int), "validation": defaultdict(int), "test": defaultdict(int)}
        for tid, split_name in id_to_split.items():
            for k in key_fn(by_id[tid]):
                out[split_name][k] += 1
        return {s: dict(d) for s, d in out.items()}

    deviation = {s: primary_split["instance_counts"][s] - primary_split["target_counts"][s]
                 for s in ("train", "validation", "test")}
    exact_match = all(v == 0 for v in deviation.values())

    return {
        "instance_counts": primary_split["instance_counts"],
        "target_counts": primary_split["target_counts"],
        "deviation_from_target": deviation,
        "exact_target_match": exact_match,
        "deviation_cause": (
            None if exact_match else
            "task_group sizes (1/2/3, from Phase 6.5B's template-based grouping) do not tile evenly into "
            "30/10/10 -- groups are never split to force an exact match (Step 6.5C's explicit priority order: "
            "group leakage 0 first, exact instance count second)."
        ),
        "template_family_distribution": _dist(lambda t: [t.task_category]),
        "difficulty_distribution": _dist(lambda t: [t.difficulty]),
        "branch_distribution": _dist(lambda t: t.expected_normal_branches),
        "service_combination_distribution": _dist(lambda t: ["+".join(sorted(t.required_services))]),
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. Secondary unseen-template generalization split
# ══════════════════════════════════════════════════════════════════════════

# [Step 6.5-12] a CONSTRAINT-COMBINATION holdout, not a whole-family holdout
# -- chosen because it's a genuinely rare combination (both budget AND
# integration conflict active) that still leaves every template_family
# represented in the secondary train/val side.
UNSEEN_CONSTRAINT_COMBINATION = ("budget_conflict", "integration_conflict")


def build_secondary_split(task_instances: List[TaskInstance]) -> Dict[str, Any]:
    def _constraint_signature(ti: TaskInstance) -> Optional[tuple]:
        branches = set(ti.expected_normal_branches)
        flags = []
        if "budget_revision" in branches:
            flags.append("budget_conflict")
        if "schedule_clarification" in branches:
            flags.append("schedule_conflict")
        if "integration_revision" in branches:
            flags.append("integration_conflict")
        return tuple(sorted(flags))

    held_out = [ti.task_instance_id for ti in task_instances if _constraint_signature(ti) == UNSEEN_CONSTRAINT_COMBINATION]
    remaining = [ti.task_instance_id for ti in task_instances if _constraint_signature(ti) != UNSEEN_CONSTRAINT_COMBINATION]
    families_in_remaining = {ti.task_category for ti in task_instances if ti.task_instance_id in remaining}
    all_families = {ti.task_category for ti in task_instances}

    return {
        "holdout_constraint_combination": list(UNSEEN_CONSTRAINT_COMBINATION),
        "test_task_ids": sorted(held_out),
        "train_and_validation_task_ids": sorted(remaining),
        "families_represented_in_train_and_validation": sorted(families_in_remaining),
        "all_families_still_represented_outside_holdout": all_families <= families_in_remaining,
        "note": "Never summed with primary_group_split results (Step 6.5-12) -- a purely exploratory "
                "generalization check.",
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. Shortcut risk validation (Step 6.5-16)
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ShortcutIssue:
    issue_code: str
    severity: str
    affected_task_ids: List[str]
    explanation: str
    recommended_fix: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def validate_shortcut_risks(task_instances: List[TaskInstance], content_bundles: Dict[str, Any],
                             primary_split: dict) -> List[ShortcutIssue]:
    issues: List[ShortcutIssue] = []
    by_id = {t.task_instance_id: t for t in task_instances}
    id_to_split = {}
    for split_name in ("train", "validation", "test"):
        for tid in primary_split[f"{split_name}_task_ids"]:
            id_to_split[tid] = split_name

    # DESTINATION_SPLIT_LEAKAGE -- a destination confined to exactly 1 split
    dest_splits: Dict[str, set] = defaultdict(set)
    for tid, split_name in id_to_split.items():
        dest_splits[by_id[tid].destination].add(split_name)
    confined = {d: s for d, s in dest_splits.items() if len(s) == 1}
    if confined:
        issues.append(ShortcutIssue(
            "DESTINATION_SPLIT_LEAKAGE", "low",
            [tid for tid, sp in id_to_split.items() if by_id[tid].destination in confined],
            f"{len(confined)} destination(s) appear in exactly one primary split: {sorted(confined)}. "
            "A detector could learn to key on destination-specific content rather than injection behavior.",
            "Acceptable at this scale (15 destinations / 50 tasks, ~3.3 tasks/destination) if the count stays "
            "small -- otherwise rebalance group->split assignment to spread multi-instance destinations."
        ))

    # BRANCH_SPLIT_CONCENTRATION -- a branch pattern (esp. multi_branch)
    # clustering heavily into one split relative to its share of instances.
    branch_by_split: Dict[str, Dict[str, int]] = {"train": defaultdict(int), "validation": defaultdict(int), "test": defaultdict(int)}
    for tid, split_name in id_to_split.items():
        for b in by_id[tid].expected_normal_branches:
            branch_by_split[split_name][b] += 1
    split_sizes = {s: sum(1 for sp in id_to_split.values() if sp == s) for s in ("train", "validation", "test")}
    concentrated = []
    for branch in {b for d in branch_by_split.values() for b in d}:
        rates = {s: branch_by_split[s].get(branch, 0) / split_sizes[s] for s in ("train", "validation", "test") if split_sizes[s]}
        if rates and max(rates.values()) - min(rates.values()) >= 0.3:
            concentrated.append((branch, rates))
    if concentrated:
        issues.append(ShortcutIssue(
            "BRANCH_SPLIT_CONCENTRATION", "low",
            [], "Branch pattern rate differs by >=30 percentage points across splits for: "
            + "; ".join(f"{b} ({', '.join(f'{s}={r:.0%}' for s, r in rates.items())})" for b, rates in concentrated)
            + f". Split sizes are small ({split_sizes}), so this is expected sampling noise at n=10 more than "
              "a designed bias, but worth a second look if it persists after Phase 6.5D.",
            "No action required at this scale unless Phase 6.5D's mock event patterns show a systematic "
            "split-dependent effect."
        ))

    # TEMPLATE_SPLIT_LEAKAGE -- a template_id split across >1 primary split
    # (should be structurally impossible since task_group_id == template
    # grouping; kept as a hard invariant check, not a soft risk)
    template_splits: Dict[str, set] = defaultdict(set)
    for tid, split_name in id_to_split.items():
        template_splits[by_id[tid].template_id].add(split_name)
    leaked_templates = {tpl: s for tpl, s in template_splits.items() if len(s) > 1}
    if leaked_templates:
        issues.append(ShortcutIssue(
            "TEMPLATE_SPLIT_LEAKAGE", "critical",
            [tid for tid, sp in id_to_split.items() if by_id[tid].template_id in leaked_templates],
            f"{len(leaked_templates)} template_id(s) have instances in more than one primary split -- "
            "this should be structurally impossible given task_group_id == template grouping.",
            "Investigate build_primary_split()/task_group_id assignment immediately -- this indicates a bug."
        ))

    # CONTENT_BUNDLE_REUSE_LEAKAGE -- a content_bundle_id shared across splits
    bundle_splits: Dict[str, set] = defaultdict(set)
    for tid, split_name in id_to_split.items():
        bundle_splits[by_id[tid].content_bundle_id].add(split_name)
    leaked_bundles = {b: s for b, s in bundle_splits.items() if len(s) > 1}
    if leaked_bundles:
        issues.append(ShortcutIssue(
            "CONTENT_BUNDLE_REUSE_LEAKAGE", "medium",
            [tid for tid, sp in id_to_split.items() if by_id[tid].content_bundle_id in leaked_bundles],
            f"{len(leaked_bundles)} content_bundle_id(s) are shared by task instances landing in different "
            "primary splits -- raw flight/hotel/tour content (prices/descriptions) could let a detector "
            "recognize train content reappearing in test.",
            "Route task_group_id assignment so every task instance sharing a content_bundle_id lands in the "
            "same split (currently NOT enforced by build_primary_split -- see generation_report follow-up)."
        ))

    # OPTION_POSITION_BIAS -- cheapest hotel option at the same array index in
    # too many bundles (content_bundle_spec.json's "no fixed cheapest-first
    # convention" requirement)
    by_bundle_hotels: Dict[str, List[dict]] = defaultdict(list)
    for h in content_bundles["hotels"]:
        tag = h["option_id"].split("_")[2]
        by_bundle_hotels[(h["destination"], tag)].append(h)
    cheapest_index_counts: Dict[int, int] = defaultdict(int)
    for opts in by_bundle_hotels.values():
        cheapest_idx = min(range(len(opts)), key=lambda i: opts[i]["total_price"])
        cheapest_index_counts[cheapest_idx] += 1
    total_bundles = sum(cheapest_index_counts.values())
    dominant_idx, dominant_count = max(cheapest_index_counts.items(), key=lambda kv: kv[1])
    if total_bundles and dominant_count / total_bundles > 0.6:
        issues.append(ShortcutIssue(
            "OPTION_POSITION_BIAS", "medium", [],
            f"The cheapest hotel option sits at array index {dominant_idx} in "
            f"{dominant_count}/{total_bundles} bundles ({dominant_count/total_bundles:.0%}).",
            "Vary the seeded generation so the cheapest option's position doesn't cluster at one index."
        ))

    # PRICE_RANK_BIAS -- cheapest hotel is ALSO always the highest-rated
    both_cheapest_and_best = 0
    for opts in by_bundle_hotels.values():
        cheapest = min(opts, key=lambda o: o["total_price"])
        best_rated = max(opts, key=lambda o: o["quality_score"])
        if cheapest["option_id"] == best_rated["option_id"]:
            both_cheapest_and_best += 1
    if total_bundles and both_cheapest_and_best / total_bundles > 0.5:
        issues.append(ShortcutIssue(
            "PRICE_RANK_BIAS", "medium", [],
            f"The cheapest hotel option is also the highest-rated option in "
            f"{both_cheapest_and_best}/{total_bundles} bundles ({both_cheapest_and_best/total_bundles:.0%}) -- "
            "violates content_bundle_spec.json's 'cheapest is not always optimal' requirement.",
            "Introduce an explicit inverse price/rating relationship in _make_hotel_options()."
        ))

    # CONTENT_LENGTH_BIAS -- not checkable yet: no attack-injected content
    # exists in a static task/content bundle (injection happens at runtime,
    # per session, via injection_builder.py) -- deferred, not silently skipped.
    issues.append(ShortcutIssue(
        "CONTENT_LENGTH_BIAS", "info", [],
        "Not checkable at Phase 6.5C -- normal vs. attack content only diverges at runtime per matched-pair "
        "session (injection_builder.py), not in this static task/content bundle.",
        "Re-check once formal attack sessions exist (Phase 6.5's formal collection, or a Step-6-style "
        "mini-validation against the formal workload)."
    ))

    # ATTACK_BRANCH_CONFOUND -- not checkable yet: no task_instance-level
    # FormalAttackApplicabilityMatrix rows exist (attack_applicability_plan.json
    # is family-level only, per its own note).
    issues.append(ShortcutIssue(
        "ATTACK_BRANCH_CONFOUND", "info", [],
        "Not checkable at Phase 6.5C -- no task_instance_id-level attack applicability rows exist yet "
        "(attack_applicability_plan.json is family-level/provisional only). All 50 task instances include "
        "'hotel' in required_services, so preference_manipulation-style attacks are not structurally confined "
        "to any single branch profile once applicability rows are built.",
        "Re-check once a FormalAttackApplicabilityMatrix with task_instance_id rows is generated."
    ))

    # NORMAL_ATTACK_CONFIG_MISMATCH -- not checkable yet, same reason as above
    issues.append(ShortcutIssue(
        "NORMAL_ATTACK_CONFIG_MISMATCH", "info", [],
        "Not checkable at Phase 6.5C -- no formal attack sessions/configs exist yet.",
        "Re-check once formal attack sessions are collected."
    ))

    # DIFFICULTY_BRANCH_CONFOUND -- not one of the original 8 shortcut codes,
    # added because Step 6.5B's generation_report already flagged the
    # underlying cause (schedule_conflict/integration_conflict are mutually
    # exclusive in content, so every hard-tier task's 2-condition requirement
    # is structurally forced to include budget_conflict).
    hard_tasks = [t for t in task_instances if t.difficulty == "hard"]
    hard_with_budget = [t for t in hard_tasks if "budget_revision" in t.expected_normal_branches]
    if hard_tasks and len(hard_with_budget) / len(hard_tasks) >= 0.9:
        issues.append(ShortcutIssue(
            "DIFFICULTY_BRANCH_CONFOUND", "medium",
            [t.task_instance_id for t in hard_with_budget],
            f"{len(hard_with_budget)}/{len(hard_tasks)} ({len(hard_with_budget)/len(hard_tasks):.0%}) hard-difficulty "
            "task instances also carry budget_revision -- difficulty='hard' and the budget_revision branch are "
            "near-perfectly correlated in this workload, a structural consequence of "
            "schedule_conflict/integration_conflict content being mutually exclusive (Step 6.5B's known_simplifications). "
            "A detector (or an evaluator reading raw event/message counts) may not be able to distinguish a "
            "'hard task' signal from a 'budget conflict' signal.",
            "If this distinction matters for the formal experiment, either relax the schedule/integration "
            "mutual-exclusion constraint (allow non-budget hard-tier constraint pairs with redesigned content) "
            "or explicitly report LightGAE results conditioned on this confound rather than treating "
            "difficulty and budget_revision as independent factors."
        ))

    return issues


# ══════════════════════════════════════════════════════════════════════════
# 4. Near-duplicate report
# ══════════════════════════════════════════════════════════════════════════


def _canonical_signature(ti: TaskInstance) -> tuple:
    def _duration_bucket(dep: str, ret: str) -> str:
        import datetime as dt
        days = (dt.date.fromisoformat(ret) - dt.date.fromisoformat(dep)).days
        if days <= 3:
            return "short"
        if days <= 5:
            return "medium"
        if days <= 7:
            return "long"
        return "extended"

    def _traveler_bucket(n: int) -> str:
        return {1: "solo", 2: "couple"}.get(n, "group")

    def _budget_bucket(amount: float) -> str:
        if amount < 800000:
            return "tight"
        if amount < 1600000:
            return "moderate"
        return "flexible"

    return (
        ti.task_category, "+".join(sorted(ti.required_services)), ti.difficulty,
        _duration_bucket(ti.departure_date, ti.return_date), _traveler_bucket(ti.travelers),
        _budget_bucket(ti.budget_amount), tuple(sorted(ti.expected_normal_branches)),
    )


def near_duplicate_report(task_instances: List[TaskInstance], primary_split: dict) -> dict:
    id_to_split = {}
    for split_name in ("train", "validation", "test"):
        for tid in primary_split[f"{split_name}_task_ids"]:
            id_to_split[tid] = split_name

    groups: Dict[tuple, List[str]] = defaultdict(list)
    for ti in task_instances:
        groups[_canonical_signature(ti)].append(ti.task_instance_id)

    near_dup_groups = {sig: ids for sig, ids in groups.items() if len(ids) > 1}
    cross_split_violations = []
    for sig, ids in near_dup_groups.items():
        splits_hit = {id_to_split[i] for i in ids}
        if len(splits_hit) > 1:
            cross_split_violations.append({"signature": list(sig), "task_ids": ids, "splits": sorted(splits_hit)})

    return {
        "near_duplicate_group_count": len(near_dup_groups),
        "near_duplicate_groups": [{"signature": list(sig), "task_ids": ids} for sig, ids in near_dup_groups.items()],
        "cross_split_violation_count": len(cross_split_violations),
        "cross_split_violations": cross_split_violations,
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. Static workload validation checklist (Step 6.5-23)
# ══════════════════════════════════════════════════════════════════════════


def validate_workload_static(task_instances: List[TaskInstance], content_bundles: Dict[str, Any],
                              primary_split: dict, shortcut_issues: List[ShortcutIssue],
                              near_dup: dict, spec_dir: str = SPEC_DIR_DEFAULT) -> dict:
    spec = load_spec(spec_dir)
    checks = {}

    checks["task_count_exactly_50"] = len(task_instances) == 50
    ids = [t.task_instance_id for t in task_instances]
    checks["task_instance_id_duplicates"] = len(ids) - len(set(ids))
    checks["task_group_id_missing_count"] = sum(1 for t in task_instances if not t.task_group_id)

    service_to_bundle = {"flight": "flights", "hotel": "hotels", "tours": "tours"}
    missing_content = 0
    for t in task_instances:
        for svc in t.required_services:
            key = service_to_bundle.get(svc)
            if key and not any(o["destination"] == t.destination for o in content_bundles[key]):
                missing_content += 1
    checks["required_service_content_missing_count"] = missing_content

    from collections import defaultdict as dd
    counts = dd(int)
    for kind, key in (("flight", "flights"), ("hotel", "hotels"), ("tour", "tours")):
        for o in content_bundles[key]:
            tag = o["option_id"].split("_")[2]
            counts[(kind, o["destination"], tag)] += 1
    minimums = {"flight": 3, "hotel": 3, "tour": 3}
    checks["option_count_below_minimum_count"] = sum(1 for k, n in counts.items() if n < minimums[k[0]])

    import datetime as dt
    checks["date_inconsistency_count"] = sum(
        1 for t in task_instances if dt.date.fromisoformat(t.return_date) <= dt.date.fromisoformat(t.departure_date))

    pairs = {c["pair"] for c in content_bundles["currency"]}
    checks["missing_currency_pair_count"] = sum(
        1 for t in task_instances if f"{t.budget_currency}/{t.target_currency}" not in pairs)

    checks["train_val_test_group_overlap_count"] = _group_overlap_count(primary_split)
    checks["near_duplicate_cross_split_count"] = near_dup["cross_split_violation_count"]

    checks["attack_applicability_coverage_note"] = (
        "family-level only (attack_applicability_plan.json) -- task_instance_id-level coverage not yet built")
    checks["hard_normal_coverage_count"] = sum(1 for t in task_instances if t.hard_normal_tags)
    checks["hard_normal_coverage_in_target_range"] = (
        spec["hard_normal_tag_taxonomy"]["coverage_target"]["minimum_tagged_task_instances"]
        <= checks["hard_normal_coverage_count"]
        <= spec["hard_normal_tag_taxonomy"]["coverage_target"]["maximum_tagged_task_instances"])

    llm_view_keys = set()
    for t in task_instances:
        llm_view_keys |= set(t.to_travel_request_kwargs().keys())
    checks["forbidden_metadata_field_leak_count"] = len(FORBIDDEN_METADATA_KEYS & llm_view_keys)

    checks["synthetic_provenance_missing_count"] = sum(
        1 for records in content_bundles.values() for r in records if r.get("source_id") != "generated_fixture")

    checks["critical_shortcut_issue_count"] = sum(1 for i in shortcut_issues if i.severity == "critical")

    checks["overall_pass"] = (
        checks["task_instance_id_duplicates"] == 0
        and checks["task_group_id_missing_count"] == 0
        and checks["required_service_content_missing_count"] == 0
        and checks["option_count_below_minimum_count"] == 0
        and checks["date_inconsistency_count"] == 0
        and checks["missing_currency_pair_count"] == 0
        and checks["train_val_test_group_overlap_count"] == 0
        and checks["near_duplicate_cross_split_count"] == 0
        and checks["hard_normal_coverage_in_target_range"]
        and checks["forbidden_metadata_field_leak_count"] == 0
        and checks["synthetic_provenance_missing_count"] == 0
        and checks["critical_shortcut_issue_count"] == 0
    )
    return checks


def _group_overlap_count(primary_split: dict) -> int:
    train, val, test = set(primary_split["train_task_ids"]), set(primary_split["validation_task_ids"]), set(primary_split["test_task_ids"])
    return len((train & val) | (train & test) | (val & test))


# ══════════════════════════════════════════════════════════════════════════
# Writers -- data/travel_a2a/formal_workload/splits/ and reports/travel_a2a/formal_workload/
# ══════════════════════════════════════════════════════════════════════════


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: List[dict]) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_splits(primary_split: dict, secondary_split: dict, balance_report: dict,
                  splits_dir: str = SPLITS_DIR_DEFAULT) -> None:
    _write_json(os.path.join(splits_dir, "primary_group_split.json"), primary_split)
    _write_json(os.path.join(splits_dir, "train_task_ids.json"), primary_split["train_task_ids"])
    _write_json(os.path.join(splits_dir, "validation_task_ids.json"), primary_split["validation_task_ids"])
    _write_json(os.path.join(splits_dir, "test_task_ids.json"), primary_split["test_task_ids"])
    _write_json(os.path.join(splits_dir, "split_balance_report.json"), balance_report)
    _write_json(os.path.join(splits_dir, "unseen_template_split.json"), secondary_split)


def write_reports(task_instances: List[TaskInstance], content_bundles: Dict[str, Any],
                   primary_split: dict, balance_report: dict, shortcut_issues: List[ShortcutIssue],
                   near_dup: dict, static_checks: dict, report_root: str = REPORT_ROOT_DEFAULT) -> None:
    workload_summary = {
        "task_count": len(task_instances),
        "template_count": len({t.template_id for t in task_instances}),
        "task_group_count": len({t.task_group_id for t in task_instances}),
        "split_unit_count": primary_split["split_unit_count"],
        "destination_count": len({t.destination for t in task_instances}),
        "content_bundle_count": len({t.content_bundle_id for t in task_instances}),
        "instance_counts_by_split": primary_split["instance_counts"],
        "overall_static_validation_pass": static_checks["overall_pass"],
    }
    _write_json(os.path.join(report_root, "workload_summary.json"), workload_summary)
    _write_csv(os.path.join(report_root, "workload_summary.csv"), [workload_summary])

    def _dist_rows(dist_by_split: Dict[str, Dict[str, int]]) -> List[dict]:
        keys = sorted({k for d in dist_by_split.values() for k in d})
        return [{"category": k, **{s: dist_by_split[s].get(k, 0) for s in ("train", "validation", "test")}}
                 for k in keys]

    _write_csv(os.path.join(report_root, "family_distribution.csv"), _dist_rows(balance_report["template_family_distribution"]))
    _write_csv(os.path.join(report_root, "difficulty_distribution.csv"), _dist_rows(balance_report["difficulty_distribution"]))
    _write_csv(os.path.join(report_root, "branch_distribution.csv"), _dist_rows(balance_report["branch_distribution"]))
    _write_csv(os.path.join(report_root, "service_combination_distribution.csv"),
               _dist_rows(balance_report["service_combination_distribution"]))

    id_to_split = {}
    for s in ("train", "validation", "test"):
        for tid in primary_split[f"{s}_task_ids"]:
            id_to_split[tid] = s
    dest_by_split: Dict[str, Dict[str, int]] = {"train": defaultdict(int), "validation": defaultdict(int), "test": defaultdict(int)}
    by_id = {t.task_instance_id: t for t in task_instances}
    for tid, s in id_to_split.items():
        dest_by_split[s][by_id[tid].destination] += 1
    _write_csv(os.path.join(report_root, "destination_distribution.csv"),
               _dist_rows({s: dict(d) for s, d in dest_by_split.items()}))

    _write_json(os.path.join(report_root, "split_balance_report.json"), balance_report)
    _write_json(os.path.join(report_root, "shortcut_risk_report.json"), [i.to_dict() for i in shortcut_issues])
    _write_json(os.path.join(report_root, "near_duplicate_report.json"), near_dup)
    _write_json(os.path.join(report_root, "validation_report.json"), {
        "phase": "6.5C_static_validation",
        "checks": static_checks,
        "note": "Mock execution (Phase 6.5D) validation is a SEPARATE report -- this covers only static "
                "generation/split/shortcut/near-duplicate checks, no Ollama/mock session was run here.",
    })


def run_phase_6_5c(output_dir: str = OUTPUT_DIR_DEFAULT, splits_dir: str = SPLITS_DIR_DEFAULT,
                    report_root: str = REPORT_ROOT_DEFAULT, spec_dir: str = SPEC_DIR_DEFAULT) -> dict:
    task_instances = load_task_instances(output_dir)
    content_bundles = load_content_bundles(output_dir)

    primary_split = build_primary_split(task_instances)
    secondary_split = build_secondary_split(task_instances)
    balance_report = build_split_balance_report(task_instances, primary_split)
    shortcut_issues = validate_shortcut_risks(task_instances, content_bundles, primary_split)
    near_dup = near_duplicate_report(task_instances, primary_split)
    static_checks = validate_workload_static(task_instances, content_bundles, primary_split,
                                              shortcut_issues, near_dup, spec_dir)

    write_splits(primary_split, secondary_split, balance_report, splits_dir)
    write_reports(task_instances, content_bundles, primary_split, balance_report,
                  shortcut_issues, near_dup, static_checks, report_root)

    return {
        "primary_split": primary_split, "secondary_split": secondary_split, "balance_report": balance_report,
        "shortcut_issues": shortcut_issues, "near_dup": near_dup, "static_checks": static_checks,
    }
