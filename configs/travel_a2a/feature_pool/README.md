# configs/travel_a2a/feature_pool

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

[Step 7] Raw metadata schema and candidate feature pool configs for the
travel_a2a_v2 formal workload. This is NOT feature engineering for LightGAE --
no threshold, AUROC, correlation-filtering, or feature-importance computation
happens anywhere under this directory. The point of Step 7 is to define, in
one auditable place, (a) every raw metadata field this environment already
produces, and (b) the candidate feature pool derived from it -- so that Step 8
onward can screen a known, documented pool rather than inventing features
ad hoc against the LightGAE input.

| file | purpose |
|---|---|
| `raw_metadata_schema.json` | Node/edge/session/runtime raw metadata field catalog, audited against the actual `models.py`/`metadata_delta.py`/`formal_workload_mock_run.py` code (Step 7A). Every field's `status` (`collected` / `field_exists_not_populated` / `planned_not_collected`) and `candidate_only`/`known_confound` annotations are grounded in what the code and Step 6.5D's validation reports actually show today -- nothing here is aspirational. |
| `candidate_feature_registry.json` | One entry per candidate feature produced by `experiments/real_llm/travel_a2a/feature_generators/` (Step 7B) plus the raw session-level aggregates from `raw_metadata_schema.json` that are themselves already usable scalars (Step 7C). Records `feature_role` (`candidate_input` / `collection_context` / `diagnostic_only`), `mock_availability` (`available_in_mock` / `requires_ollama_runtime` / `available_after_normal_statistics`), `missing_value_policy`, `leakage_risk`, and cross-references (`mathematically_dependent_on`, `potentially_redundant_with`) for later redundancy screening. Does not screen, score, or select any feature. |

See also `reports/travel_a2a/formal_workload_dataset_card.md` (Step 6.5) for
the difficulty confounds (`event_count`/`message_count`/`revision_count`)
this schema's `candidate_only`/`known_confound` fields reference directly, and
`reports/travel_a2a/feature_pool/feature_generation_manifest.json` (Step 7B)
for the empirical null-rate evidence behind each feature's `mock_availability`.
