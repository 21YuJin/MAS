# data/travel_a2a/development

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

The 6 hand-authored task fixtures from Steps 1-6 (`tasks/normal_tasks.json`)
and their matching external content (`content/{flights,hotels,currency,tours}.json`).

## Role

- Smoke tests (`experiments/real_llm/tests/test_travel_a2a_workflow.py`)
- Workflow regression
- Evaluator unit/integration tests (`test_travel_a2a_attacks.py`, `test_travel_a2a_step6.py`)
- Attack scenario development and debugging (Steps 5-6)

## Boundary

These 6 fixtures are **never** used as the formal LightGAE dataset (see
`../formal_workload/`). `fixtures.py`'s `DEFAULT_TASKS_PATH` and
`content_repository.py`'s `DEFAULT_CONTENT_DIR` point here by default; formal
workload code uses its own separate loader and never falls back to this path.
