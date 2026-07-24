"""
[Phase 7D] Builds configs/travel_a2a/feature_pool/screening_plan.json -- a
STATIC PLAN for Step 8's screening, not an execution of it. No feature is
scored, correlated, fit, or removed anywhere in this module. status is always
PLAN_ONLY_NOT_EXECUTED.
"""
import json
import os
from typing import Any, Dict, List

_STAGES: List[Dict[str, Any]] = [
    {"screening_stage": 1, "criterion": "schema_quality",
     "input_scope": "all registered features",
     "data_source": "candidate_feature_registry.json",
     "threshold": "registry_validator.validate_registry_integrity(registry)['passed'] is True",
     "action": "block screening until registry integrity passes", "manual_review_required": False},
    {"screening_stage": 2, "criterion": "missingness",
     "input_scope": "features with mock_availability in {available_in_mock, requires_ollama_runtime}",
     "data_source": "feature_generation_manifest.json null_rate_by_field (mock) + a future ollama_runtime manifest",
     "threshold": "null_rate > 0.5 on the backend that SHOULD make this feature available",
     "action": "flag for review, do not auto-remove", "manual_review_required": True},
    {"screening_stage": 3, "criterion": "constant_and_low_variance",
     "input_scope": "normal TRAIN split sessions only",
     "data_source": "future ollama_runtime formal collection (train split)",
     "threshold": "variance == 0 or coefficient_of_variation < 0.01 across normal train sessions",
     "action": "flag as low-information, do not auto-remove", "manual_review_required": True},
    {"screening_stage": 4, "criterion": "mathematical_redundancy",
     "input_scope": "redundancy_groups.json members",
     "data_source": "candidate_feature_registry.json (mathematically_dependent_on / potentially_redundant_with)",
     "threshold": "group size > 1",
     "action": "prefer recommended_representative per group; keep all members as candidates, do not auto-remove",
     "manual_review_required": True},
    {"screening_stage": 5, "criterion": "empirical_correlation",
     "input_scope": "normal TRAIN split sessions only, enabled candidate_input features",
     "data_source": "future ollama_runtime formal collection (train split)",
     "threshold": "Pearson |r| > 0.95 or Spearman |rho| > 0.95",
     "action": "prefer one representative per correlated pair, do not auto-remove", "manual_review_required": True},
    {"screening_stage": 6, "criterion": "normal_stability",
     "input_scope": "normal sessions, stratified by agent / task_family / difficulty / split / repeat",
     "data_source": "future ollama_runtime formal collection",
     "threshold": "high variance across strata not attributable to genuine behavioral diversity",
     "action": "flag for review -- difficulty is a STRATIFICATION variable here, never a feature input",
     "manual_review_required": True},
    {"screening_stage": 7, "criterion": "environment_sensitivity",
     "input_scope": "timing/duration features",
     "data_source": "future ollama_runtime formal collection, repeated runs",
     "threshold": "large variation attributable to warm-up, run order, or hardware backend",
     "action": "flag for review or require warm-up-controlled collection", "manual_review_required": True},
    {"screening_stage": 8, "criterion": "deployment_feasibility",
     "input_scope": "all enabled candidate_input features",
     "data_source": "deployment_feasibility_report.csv",
     "threshold": "deployment_category == offline_only outside a controlled research setting",
     "action": "flag as research-only if not deployment_available", "manual_review_required": False},
    {"screening_stage": 9, "criterion": "feature_family_ablation",
     "input_scope": "enabled candidate_input features grouped by feature_family",
     "data_source": "future LightGAE ablation runs (normal + attack VALIDATION split only)",
     "threshold": "n/a -- exploratory, not a pass/fail gate",
     "action": "inform Reduced Core Set discussion, do not decide it here", "manual_review_required": True},
    {"screening_stage": 10, "criterion": "reduced_core_selection",
     "input_scope": "surviving features from stages 1-9",
     "data_source": "human decision informed by stages 1-9's reports",
     "threshold": "n/a -- final selection is a human decision, not an automated threshold",
     "action": "NOT PERFORMED in Step 7 or in this plan -- reserved for a later step once formal Ollama attack data exists",
     "manual_review_required": True},
]

_RULES = {
    "attack_data_usage_rule": (
        "Formal attack-condition sessions/results must NEVER be used to remove or select a feature at any stage "
        "above -- stages 3/5/6/7 read normal-condition sessions only (train split). Attack data is reserved for "
        "final LightGAE evaluation (a later step), never for feature screening."
    ),
    "difficulty_usage_rule": (
        "difficulty is used ONLY as a stratification/grouping variable in stage 6, never as a feature value fed "
        "to a model -- consistent with raw_metadata_schema.json/candidate_feature_registry.json excluding it as a "
        "source field anywhere."
    ),
}


def build_screening_plan() -> Dict[str, Any]:
    return {
        "experiment_version": "travel_a2a_v2",
        "step": "Step 7D",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "stages": _STAGES,
        "rules": _RULES,
    }


def write_screening_plan(config_root: str) -> Dict[str, Any]:
    plan = build_screening_plan()
    os.makedirs(config_root, exist_ok=True)
    with open(os.path.join(config_root, "screening_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan
