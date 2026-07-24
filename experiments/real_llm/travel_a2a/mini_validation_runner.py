"""
[Step 6] Mini-validation orchestrator -- runs Phase 6A (evaluator dry run,
no Ollama calls), Phase 6B (small repeated run), and Phase 6C (fuller
mini-validation), aggregating attack statistics, metadata deltas, and a
manual review queue, per the Step 6 instruction. Never filters sessions by
success -- every injection-attempted session stays in every aggregate,
regardless of goal_success (Step 6's own restated principle).
"""
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from .applicability import build_attack_config_for_task, find_row, load_applicability_matrix
from .attack_evaluators import evaluate_attack
from .attack_models import AttackConfig
from .content_repository import ContentRepository, load_content_repository
from .fixtures import load_task_fixture_dicts
from .matched_pair_runner import DEFAULT_ATTACK_SMOKE_ROOT, MatchedPairRunner, save_matched_pair
from .metadata_delta import classify_outcome_group, compute_metadata_delta_summary
from .mock_runner import MockWorkflowResult
from .session_store import load_session

DEFAULT_STEP6_OUTPUT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "travel_a2a", "step6_mini_validation"))
DEFAULT_STEP6_REPORT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reports", "travel_a2a", "step6"))

ATTACK_CONFIG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "travel_a2a", "attacks"))

_STAT_KEYS = ("entry_agent_exposed", "instruction_followed", "indicator_observed", "artifact_changed",
              "propagation_observed", "workflow_changed", "goal_success")


def load_attack_template(attack_family: str, payload_variant_id: str = "v1") -> AttackConfig:
    path = os.path.join(ATTACK_CONFIG_DIR, f"{attack_family}_{payload_variant_id}.json")
    with open(path, encoding="utf-8") as f:
        return AttackConfig.from_dict(json.load(f))


class RunIndexTracker:
    """[Step 6-5/6-15] Simple monotonic counters -- run_index_global across
    the whole mini-validation session, run_index_for_task/run_index_for_attack
    scoped to a specific fixture / attack_family respectively."""

    def __init__(self):
        self.global_index = 0
        self.per_task: Dict[str, int] = defaultdict(int)
        self.per_attack: Dict[str, int] = defaultdict(int)

    def next(self, task_fixture_id: str, attack_family: str):
        idx = (self.global_index, self.per_task[task_fixture_id], self.per_attack[attack_family])
        self.global_index += 1
        self.per_task[task_fixture_id] += 1
        self.per_attack[attack_family] += 1
        return idx


def run_phase(plan: List[dict], content_repository: Optional[ContentRepository] = None,
              output_root: str = DEFAULT_STEP6_OUTPUT_ROOT, hardware_backend: Optional[str] = None,
              save_sessions: bool = True) -> List[dict]:
    """
    plan: list of {"attack_family", "payload_variant_id", "task_fixture_id",
    "repeat_index", "order_seed"}. Each entry is applicability-checked before
    running (InapplicableCombinationError propagates -- a caller building a
    plan from get_applicable_tasks() should never hit this).

    Returns a list of result dicts: {pair_result, metadata_delta,
    outcome_group} -- normal_result/attack_result are saved to disk (Step
    6-12/5-12 layout) but not kept in memory across the whole phase, since a
    54-72 session phase would otherwise hold everything at once.
    """
    content_repository = content_repository or load_content_repository()
    fixtures = {f["task_fixture_id"]: f for f in load_task_fixture_dicts()}
    matrix = load_applicability_matrix()
    runner = MatchedPairRunner(content_repository)
    tracker = RunIndexTracker()
    results = []

    for entry in plan:
        template = load_attack_template(entry["attack_family"], entry.get("payload_variant_id", "v1"))
        attack_config = build_attack_config_for_task(template, matrix, entry["task_fixture_id"])
        run_idx_global, run_idx_task, run_idx_attack = tracker.next(entry["task_fixture_id"], entry["attack_family"])

        pair_result, normal_result, attack_result = runner.run(
            fixtures[entry["task_fixture_id"]], attack_config,
            repeat_index=entry.get("repeat_index", 0), order_seed=entry.get("order_seed"),
            run_index_global=run_idx_global, run_index_for_task=run_idx_task, run_index_for_attack=run_idx_attack,
            hardware_backend=hardware_backend)

        metadata_delta = compute_metadata_delta_summary(normal_result, attack_result, pair_result.attack_diagnostics)

        if save_sessions:
            save_matched_pair(pair_result, normal_result, attack_result, attack_config, output_root=output_root)

        results.append({
            "pair_result": pair_result.to_dict(),
            "attack_config": attack_config.to_dict(),
            "metadata_delta": metadata_delta,
        })
        print(f"  [{run_idx_global+1}] {entry['task_fixture_id']:<28} {entry['attack_family']:<24} "
              f"variant={entry.get('payload_variant_id','v1')} rep={entry.get('repeat_index',0)} "
              f"-> goal_success={pair_result.attack_diagnostics['goal_success']} "
              f"outcome={metadata_delta['outcome_group']}", flush=True)

    return results


def build_plan(matrix, families_and_task_counts: Dict[str, int], repeats: int,
               variants: List[str], order_seed_base: int = 1000) -> List[dict]:
    """families_and_task_counts: {attack_family: n_tasks} -- takes the first
    n_tasks applicable rows for that family, in matrix file order (stable,
    reproducible task selection, not random)."""
    plan = []
    for family, n_tasks in families_and_task_counts.items():
        rows = [r for r in matrix if r.attack_family == family and r.applicable][:n_tasks]
        for row in rows:
            for variant in variants:
                for repeat_index in range(repeats):
                    plan.append({
                        "attack_family": family, "payload_variant_id": variant,
                        "task_fixture_id": row.task_fixture_id, "repeat_index": repeat_index,
                        "order_seed": order_seed_base + repeat_index,
                    })
    return plan


# ══════════════════════════════════════════════════════════════════════════
# Phase 6A -- evaluator dry run against already-saved sessions, no Ollama
# ══════════════════════════════════════════════════════════════════════════


def _load_as_workflow_result(session_dir_name: str, pair_dir: str) -> MockWorkflowResult:
    data = load_session(session_dir_name, output_root=pair_dir)
    return MockWorkflowResult(task=data["task"], messages=data["messages"], parts=data["parts"],
                               artifacts=data["artifacts"], events=data["events"],
                               agent_call_records=data["agent_call_records"], status_transition_issues=[])


def run_phase_6a_dry_run(attack_smoke_root: str = DEFAULT_ATTACK_SMOKE_ROOT) -> List[dict]:
    """Re-runs evaluate_attack() against every already-saved pair under
    attack_smoke_root (Step 5's 3 real sessions, by default) -- zero Ollama
    calls. Confirms: (a) the evaluator still runs cleanly after Step 6's
    calibration changes, (b) re-evaluating the SAME saved raw data twice
    yields the SAME core diagnostic values (reproducibility of the evaluator
    itself, not of the LLM)."""
    results = []
    if not os.path.isdir(attack_smoke_root):
        return results
    for pair_dir_name in sorted(os.listdir(attack_smoke_root)):
        pair_dir = os.path.join(attack_smoke_root, pair_dir_name)
        if not os.path.isdir(pair_dir):
            continue
        with open(os.path.join(pair_dir, "attack_config.json"), encoding="utf-8") as f:
            attack_config = AttackConfig.from_dict(json.load(f))
        with open(os.path.join(pair_dir, "matched_pair_result.json"), encoding="utf-8") as f:
            original_pair_result = json.load(f)

        normal_result = _load_as_workflow_result("normal", pair_dir)
        attack_result = _load_as_workflow_result("attack", pair_dir)
        diagnostics = evaluate_attack(attack_config, normal_result, attack_result,
                                       session_id=original_pair_result["attack_session_id"])
        original_diag = original_pair_result["attack_diagnostics"]
        reproducible = all(diagnostics.to_dict().get(k) == original_diag.get(k)
                            for k in ("goal_success", "indicator_observed", "propagation_observed", "workflow_changed"))
        metadata_delta = compute_metadata_delta_summary(normal_result, attack_result, diagnostics.to_dict())
        results.append({
            "pair_result": {**original_pair_result, "attack_diagnostics": diagnostics.to_dict()},
            "attack_config": attack_config.to_dict(),
            "metadata_delta": metadata_delta,
            "reproducible_vs_original": reproducible,
        })
        print(f"  [dry-run] {pair_dir_name:<55} reproducible={reproducible}  "
              f"goal_success={diagnostics.goal_success}", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════
# Aggregation / report writers
# ══════════════════════════════════════════════════════════════════════════


def aggregate_attack_summary(results: List[dict]) -> List[dict]:
    """[Step 6-9] Per (attack_family, task_fixture_id, payload_variant_id)
    rate statistics -- exploratory only (Step 6-9: no significance tests)."""
    groups = defaultdict(list)
    for r in results:
        cfg = r["attack_config"]
        key = (cfg["attack_family"], r["pair_result"]["task_fixture_id"], cfg["payload_variant_id"])
        groups[key].append(r["pair_result"]["attack_diagnostics"])

    rows = []
    for (family, task, variant), diags in sorted(groups.items()):
        n = len(diags)
        row = {"attack_family": family, "task_fixture_id": task, "payload_variant_id": variant, "attempt_count": n}
        for key in _STAT_KEYS:
            row[f"{key}_rate"] = sum(1 for d in diags if d.get(key)) / n if n else None
        row["mean_propagation_depth"] = (sum(d.get("propagation_depth", 0) for d in diags) / n) if n else None
        row["evaluator_error_rate"] = sum(1 for d in diags if d.get("evaluator_error")) / n if n else None
        rows.append(row)
    return rows


def write_attack_summary(rows: List[dict], report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> None:
    os.makedirs(report_root, exist_ok=True)
    with open(os.path.join(report_root, "attack_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if rows:
        with open(os.path.join(report_root, "attack_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_task_attack_matrix(matrix, report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> None:
    os.makedirs(report_root, exist_ok=True)
    with open(os.path.join(report_root, "task_attack_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack_family", "task_fixture_id", "entry_agent_id",
                                                "injection_source_id", "evaluator_id", "applicable", "exclusion_reason"])
        writer.writeheader()
        for row in matrix:
            writer.writerow(row.to_dict())


def write_manual_review_queue(results: List[dict], report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> int:
    """[Step 6-8] Samples: every manual_review_required=True session, plus
    every evaluator_error, plus (per family) up to 2 goal_success=False
    sessions so a reviewer also sees "nothing happened" cases, not only
    flagged ones."""
    os.makedirs(report_root, exist_ok=True)
    rows = []
    failure_samples_by_family: Dict[str, int] = defaultdict(int)
    for r in results:
        d = r["pair_result"]["attack_diagnostics"]
        cfg = r["attack_config"]
        include = False
        if d.get("manual_review_required") or d.get("evaluator_error"):
            include = True
        elif not d.get("goal_success") and failure_samples_by_family[cfg["attack_family"]] < 2:
            include = True
            failure_samples_by_family[cfg["attack_family"]] += 1
        if include:
            rows.append({
                "review_id": f"review_{len(rows)+1:04d}",
                "session_id": r["pair_result"]["attack_session_id"],
                "pair_id": r["pair_result"]["pair_id"],
                "attack_family": cfg["attack_family"],
                "evaluator_result": ("goal_success" if d.get("goal_success") else
                                      "propagated" if d.get("propagation_observed") else
                                      "entry_effect_only" if (d.get("instruction_followed") or d.get("indicator_observed")) else
                                      "no_effect"),
                "evidence_summary": "; ".join(d.get("manual_review_reasons", [])) or "sampled_failure_case",
                "reviewer_label": "", "reviewer_notes": "",
            })
    if rows:
        with open(os.path.join(report_root, "manual_review_queue.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def write_metadata_delta_summary(results: List[dict], report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> None:
    os.makedirs(report_root, exist_ok=True)
    rows = []
    for r in results:
        cfg = r["attack_config"]
        md = r["metadata_delta"]
        for delta_key, delta_val in md["deltas"].items():
            rows.append({
                "pair_id": r["pair_result"]["pair_id"], "attack_family": cfg["attack_family"],
                "task_fixture_id": r["pair_result"]["task_fixture_id"], "outcome_group": md["outcome_group"],
                "metric": delta_key, "normal_value": delta_val["normal_value"],
                "attack_value": delta_val["attack_value"], "absolute_delta": delta_val["absolute_delta"],
                "relative_delta": delta_val["relative_delta"],
            })
    if rows:
        with open(os.path.join(report_root, "metadata_delta_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_propagation_trace_summary(results: List[dict], report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> None:
    os.makedirs(report_root, exist_ok=True)
    traces = [{"pair_id": r["pair_result"]["pair_id"], "task_fixture_id": r["pair_result"]["task_fixture_id"],
               "hop_trace": r["pair_result"]["attack_diagnostics"]["hop_trace"]}
              for r in results if r["attack_config"]["attack_family"] == "downstream_propagation"]
    with open(os.path.join(report_root, "propagation_trace_summary.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


def write_manifest(phase_name: str, results: List[dict], report_root: str = DEFAULT_STEP6_REPORT_ROOT) -> None:
    os.makedirs(report_root, exist_ok=True)
    manifest_path = os.path.join(report_root, "mini_validation_manifest.json")
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.append({
        "phase": phase_name, "session_count": len(results) * 2,
        "pair_count": len(results),
        "evaluator_error_count": sum(1 for r in results if r["pair_result"]["attack_diagnostics"].get("evaluator_error")),
        "request_hash_mismatch_count": sum(1 for r in results if not r["pair_result"]["request_hash_equal"]),
        "base_content_hash_mismatch_count": sum(1 for r in results if not r["pair_result"]["base_content_hash_equal"]),
    })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
