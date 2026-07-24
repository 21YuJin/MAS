# configs/travel_a2a/formal_workload

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

[Step 6.5] Specification/policy configs for the 50-task formal synthetic
workload -- consumed by `FormalWorkloadGenerator` (Step 6.5B) to materialize
`data/travel_a2a/formal_workload/task_instances/` and `content/`. Nothing
here is itself a generated task instance; these files are the SPEC the
generator must satisfy, checked again after generation by
`validate_shortcut_risks()` / `near_duplicate_report` (Step 6.5C).

| file | purpose |
|---|---|
| `task_family_spec.json` | 7 template families, target task count, target difficulty breakdown per family (Step 6.5-4) |
| `difficulty_criteria.json` | non-length-based easy/medium/hard criteria (Step 6.5-5) |
| `destination_catalog.json` | >=12 destinations, origin catalog, trip-duration/traveler/budget-level/service-combination buckets (Step 6.5-6) |
| `branch_distribution_target.json` | target normal-branch pattern counts, to avoid a single dominant event sequence (Step 6.5-9) |
| `content_bundle_spec.json` | minimum option counts and no-shortcut trade-off requirements per content bundle (Step 6.5-7) |
| `split_policy.json` | primary group-aware 30/10/10 split + secondary unseen-template generalization split (Step 6.5-10..12) |
| `hard_normal_tag_taxonomy.json` | controlled vocabulary + target coverage for attack-free but metadata-heavy normal tasks (Step 6.5-15) |
| `attack_applicability_plan.json` | family-level provisional attack-candidate status (Step 6.5-13/6.5-14) |
| `dataset_policy.json` | label/matched-pair/attack-failure/split-unit/leakage-prevention policy (Step 6.5-18) |
| `formal_collection_plan.json` | not-yet-executed collection plan manifest (Step 6.5-19) |
| `manual_review_plan.json` | not-yet-executed manual-review sampling plan for the future formal attack collection (Step 6.5-22) |

See also `reports/travel_a2a/formal_workload_dataset_card.md` (Step 6.5-20)
and `reports/travel_a2a/benchmark_taxonomy_mapping.md` (Step 6.5-21).
