# Phase 7D Summary -- Registry-Based Feature Validation and Screening Preparation

```
experiment_version: travel_a2a_v2
step:               Step 7D
```

No feature was removed, no Reduced Core Set was selected, no normal statistics were fit, no correlation was computed, and no LightGAE training happened in this phase -- this document only answers what is already known from the registry, the raw schema, and the mock feature_generation_manifest.

## What is computable right now?
- 31 of 52 features are `available_now_mock` (graph/session-structural features and wall_clock_latency_ms-based ones -- no Ollama needed).

## What needs an Ollama run first?
- 20 features are `requires_ollama_runtime` (every token/timing feature derived from prompt_eval_count/eval_count/*_duration) -- still PLANNED_NOT_EXECUTED (configs/travel_a2a/formal_workload/formal_collection_plan.json).

## What needs normal-train statistics?
- 1 feature (`normalization_features.agent_zscore`) -- disabled until a train-split-only fit exists (Step 8+).

## How are duplicates/redundancies grouped?
- 8 redundancy group(s) found via registry-declared `mathematically_dependent_on`/`potentially_redundant_with` relationships. Every group's `auto_remove` is `false` -- only a `recommended_representative` is suggested.

## Is there any leakage risk?
- Registry-text leakage scan: PASSED, no forbidden/provenance term found in any formula/source_fields.

## What confounds are known?
- 3 CONFIRMED confound(s) (Phase 6.5D evidence): event_count, message_count, revision_count.
- 6 additional WARNING-only watch feature(s), not auto-flagged.

## What's deployable in a real system?
- See `deployment_feasibility_report.csv` -- graph/session-structural features are `portable`; token/timing features are `ollama_specific`; `agent_zscore` is `offline_only` (needs a fit stage); runtime-context fields are `diagnostic_only` (never a model input).

## Registry integrity
- `validate_registry_integrity`: PASSED (0 duplicate names, 0 missing keys, 0 dangling refs, 0 cyclic dependencies).

## What order will Step 8 screen in?
1. schema_quality  2. missingness  3. constant_and_low_variance  4. mathematical_redundancy  5. empirical_correlation  6. normal_stability  7. environment_sensitivity  8. deployment_feasibility  9. feature_family_ablation  10. reduced_core_selection -- see `configs/travel_a2a/feature_pool/screening_plan.json` (status: PLAN_ONLY_NOT_EXECUTED). Attack-condition data is explicitly excluded from every stage above by the plan's `attack_data_usage_rule`.
