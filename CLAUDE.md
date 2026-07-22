# Repository policy

## Git branches

- `v1` — frozen pilot reference (original 4-agent research pipeline: Orchestrator
  → Researcher → Analyst → Writer). Never modify. Pushed to `origin/v1`.
- `main` — active development line. This is where the `travel_a2a_v2`
  A2A-inspired travel-booking framework and its new collection structure are
  built. **Do not create a separate `v2` (or any other) git branch for this
  work** — experiment generations are distinguished by version metadata and
  directory namespacing (below), not by branches.

## Experiment version metadata

Every config/cache/dataset/output produced for the new framework carries:

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

`travel_a2a_v2` is a dataset/experiment-generation label, not a git branch name.

## Directory layout

New, parallel top-level paths for the travel_a2a generation — each contains a
short README stating its purpose and the version metadata above:

- `configs/travel_a2a/`
- `data/travel_a2a/`
- `cache/travel_a2a/`
- `outputs/travel_a2a/` (plural — distinct from the existing singular `output/`)
- `artifacts/travel_a2a/`

Existing pilot paths (`configs/attacks/`, `data/tasks/`, `output/real_llm/`,
`experiments/real_llm/`) are untouched by this work:

- Never delete or overwrite files under these existing paths.
- Never auto-migrate data from an existing path into a `travel_a2a/` path.
- If a new script needs something equivalent to an old file, write a new
  file under the `travel_a2a` path instead of editing the old one in place.
