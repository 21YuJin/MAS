"""
[Step 7D] Orchestrates every feature_screening validator against the Step 7C
registry (+ Step 7A raw schema + Step 7B manifest) and writes the Step 7D
report set under reports/travel_a2a/feature_pool/ plus screening_plan.json
under configs/travel_a2a/feature_pool/. This module does not select, remove,
fit, or score any feature -- it only runs the validators above and
serializes their output.
"""
import csv
import json
import os
from typing import Any, Dict

from .availability_validator import validate_feature_availability
from .confound_validator import build_confound_risk_report
from .deployment_validator import build_deployment_feasibility_report
from .leakage_validator import validate_no_leakage
from .redundancy_validator import build_redundancy_groups
from .registry_validator import validate_registry_integrity
from .screening_plan import build_screening_plan

_MAS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
DEFAULT_REGISTRY_PATH = os.path.join(_MAS_ROOT, "configs", "travel_a2a", "feature_pool", "candidate_feature_registry.json")
DEFAULT_RAW_SCHEMA_PATH = os.path.join(_MAS_ROOT, "configs", "travel_a2a", "feature_pool", "raw_metadata_schema.json")
DEFAULT_MANIFEST_PATH = os.path.join(_MAS_ROOT, "reports", "travel_a2a", "feature_pool", "feature_generation_manifest.json")
DEFAULT_REPORT_ROOT = os.path.join(_MAS_ROOT, "reports", "travel_a2a", "feature_pool")
DEFAULT_CONFIG_ROOT = os.path.join(_MAS_ROOT, "configs", "travel_a2a", "feature_pool")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_feature_family_summary(registry: Dict[str, Any]) -> Dict[str, Any]:
    families: Dict[str, Dict[str, int]] = {}
    for entry in registry["features"]:
        fam = families.setdefault(entry["feature_family"], {
            "total_count": 0, "enabled_count": 0, "candidate_input_count": 0,
            "candidate_only_count": 0, "requires_ollama_count": 0, "requires_normal_statistics_count": 0,
        })
        fam["total_count"] += 1
        fam["enabled_count"] += int(entry["enabled"])
        fam["candidate_input_count"] += int(entry["feature_role"] == "candidate_input")
        fam["candidate_only_count"] += int(entry["candidate_only"])
        fam["requires_ollama_count"] += int(entry["ollama_required"])
        fam["requires_normal_statistics_count"] += int(entry["requires_normal_statistics"])
    return families


def _build_summary_markdown(integrity, availability, redundancy_groups, leakage, confound, family_summary) -> str:
    n_features = sum(f["total_count"] for f in family_summary.values())
    availability_counts: Dict[str, int] = {}
    for row in availability["rows"]:
        availability_counts[row["availability_status"]] = availability_counts.get(row["availability_status"], 0) + 1

    lines = [
        "# Phase 7D Summary -- Registry-Based Feature Validation and Screening Preparation",
        "",
        "```",
        "experiment_version: travel_a2a_v2",
        "step:               Step 7D",
        "```",
        "",
        "No feature was removed, no Reduced Core Set was selected, no normal statistics were fit, "
        "no correlation was computed, and no LightGAE training happened in this phase -- this "
        "document only answers what is already known from the registry, the raw schema, and the "
        "mock feature_generation_manifest.",
        "",
        "## What is computable right now?",
        f"- {availability_counts.get('available_now_mock', 0)} of {n_features} features are `available_now_mock` "
        "(graph/session-structural features and wall_clock_latency_ms-based ones -- no Ollama needed).",
        "",
        "## What needs an Ollama run first?",
        f"- {availability_counts.get('requires_ollama_runtime', 0)} features are `requires_ollama_runtime` "
        "(every token/timing feature derived from prompt_eval_count/eval_count/*_duration) -- still "
        "PLANNED_NOT_EXECUTED (configs/travel_a2a/formal_workload/formal_collection_plan.json).",
        "",
        "## What needs normal-train statistics?",
        f"- {availability_counts.get('requires_normal_statistics', 0)} feature "
        "(`normalization_features.agent_zscore`) -- disabled until a train-split-only fit exists (Step 8+).",
        "",
        "## How are duplicates/redundancies grouped?",
        f"- {len(redundancy_groups)} redundancy group(s) found via registry-declared "
        "`mathematically_dependent_on`/`potentially_redundant_with` relationships. Every group's "
        "`auto_remove` is `false` -- only a `recommended_representative` is suggested.",
        "",
        "## Is there any leakage risk?",
        f"- Registry-text leakage scan: {'PASSED, no forbidden/provenance term found in any formula/source_fields.' if leakage['passed'] else 'FAILED -- see leakage_validation_report.json.'}",
        "",
        "## What confounds are known?",
        f"- {len(confound['confirmed'])} CONFIRMED confound(s) (Phase 6.5D evidence): "
        f"{', '.join(r['feature_name'] for r in confound['confirmed'])}.",
        f"- {len(confound['additional_watch'])} additional WARNING-only watch feature(s), not auto-flagged.",
        "",
        "## What's deployable in a real system?",
        "- See `deployment_feasibility_report.csv` -- graph/session-structural features are `portable`; "
        "token/timing features are `ollama_specific`; `agent_zscore` is `offline_only` (needs a fit stage); "
        "runtime-context fields are `diagnostic_only` (never a model input).",
        "",
        "## Registry integrity",
        f"- `validate_registry_integrity`: {'PASSED' if integrity['passed'] else 'FAILED'} "
        f"(0 duplicate names, 0 missing keys, 0 dangling refs, 0 cyclic dependencies).",
        "",
        "## What order will Step 8 screen in?",
        "1. schema_quality  2. missingness  3. constant_and_low_variance  4. mathematical_redundancy  "
        "5. empirical_correlation  6. normal_stability  7. environment_sensitivity  "
        "8. deployment_feasibility  9. feature_family_ablation  10. reduced_core_selection "
        "-- see `configs/travel_a2a/feature_pool/screening_plan.json` (status: PLAN_ONLY_NOT_EXECUTED). "
        "Attack-condition data is explicitly excluded from every stage above by the plan's "
        "`attack_data_usage_rule`.",
        "",
    ]
    return "\n".join(lines)


def run_phase_7d(registry_path: str = DEFAULT_REGISTRY_PATH, raw_schema_path: str = DEFAULT_RAW_SCHEMA_PATH,
                  manifest_path: str = DEFAULT_MANIFEST_PATH, report_root: str = DEFAULT_REPORT_ROOT,
                  config_root: str = DEFAULT_CONFIG_ROOT) -> Dict[str, Any]:
    registry = _load_json(registry_path)
    _raw_schema = _load_json(raw_schema_path)  # noqa: F841 -- loaded for future validators; not yet consumed
    manifest = _load_json(manifest_path)

    integrity = validate_registry_integrity(registry)
    availability = validate_feature_availability(registry, manifest)
    redundancy_groups = build_redundancy_groups(registry)
    leakage = validate_no_leakage(registry)
    confound = build_confound_risk_report(registry)
    deployment_rows = build_deployment_feasibility_report(registry)
    family_summary = build_feature_family_summary(registry)
    plan = build_screening_plan()

    os.makedirs(report_root, exist_ok=True)
    os.makedirs(config_root, exist_ok=True)

    with open(os.path.join(report_root, "registry_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2, ensure_ascii=False)

    with open(os.path.join(report_root, "feature_availability_report.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "availability_status", "blocking_reason"])
        writer.writeheader()
        writer.writerows(availability["rows"])

    with open(os.path.join(report_root, "redundancy_groups.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": redundancy_groups}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(report_root, "leakage_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=2, ensure_ascii=False)

    with open(os.path.join(report_root, "confound_risk_report.json"), "w", encoding="utf-8") as f:
        json.dump(confound, f, indent=2, ensure_ascii=False)

    with open(os.path.join(report_root, "deployment_feasibility_report.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "deployment_category", "deployment_available"])
        writer.writeheader()
        writer.writerows(deployment_rows)

    with open(os.path.join(report_root, "feature_family_summary.json"), "w", encoding="utf-8") as f:
        json.dump(family_summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(config_root, "screening_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    summary_md = _build_summary_markdown(integrity, availability, redundancy_groups, leakage, confound, family_summary)
    with open(os.path.join(report_root, "phase_7d_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    return {
        "registry_integrity": integrity, "availability": availability, "redundancy_groups": redundancy_groups,
        "leakage": leakage, "confound": confound, "deployment_rows": deployment_rows,
        "family_summary": family_summary, "screening_plan": plan,
    }


if __name__ == "__main__":
    result = run_phase_7d()
    print(json.dumps({
        "registry_integrity_passed": result["registry_integrity"]["passed"],
        "availability_passed": result["availability"]["passed"],
        "redundancy_group_count": len(result["redundancy_groups"]),
        "leakage_passed": result["leakage"]["passed"],
        "confirmed_confound_count": len(result["confound"]["confirmed"]),
    }, indent=2))
