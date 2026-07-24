# data/travel_a2a

Task/dataset definitions for the `travel_a2a_v2` experiment generation
(A2A-inspired travel-booking multi-agent scenario).

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

Independent of `data/tasks/` (real_llm pilot, unchanged -- see `v1` branch and
`experiments/real_llm/`). Nothing here overwrites or migrates that path.

## Layout (Step 6.5)

```
data/travel_a2a/
├── development/     -- the original 6 hand-authored task fixtures (Steps 1-6):
│                       smoke tests, workflow regression, evaluator unit
│                       tests, attack development. Never used as the formal
│                       LightGAE dataset.
└── formal_workload/  -- the 50-task formal synthetic workload (Step 6.5
                          onward): task_templates/, task_instances/, content/,
                          manifests/, splits/. See formal_workload/README.md
                          and reports/travel_a2a/formal_workload_dataset_card.md.
```

`development/` and `formal_workload/` are never mixed: no formal_workload
script reads from `development/`, and no development-fixture test reads from
`formal_workload/`.
