# data/travel_a2a/formal_workload

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

[Step 6.5] The 50-task formal synthetic travel workload used for LightGAE's
actual detection experiments -- distinct from `../development/`'s 6
hand-authored fixtures, which stay reserved for regression/attack-development
use and are never mixed in here.

## Layout

```
formal_workload/
├── task_templates/   -- TaskTemplate JSON: task STRUCTURE and constraint
│                        type, independent of any specific destination/date/
│                        budget (Step 6.5-3).
├── task_instances/    -- TaskInstance JSON: the 50 frozen, parameter-filled
│                        instances (destination/dates/budget/travelers/
│                        preferences), each referencing a template_id and a
│                        content_bundle_id (Step 6.5-3/6.5-8).
├── content/           -- source-controlled external content bundles (flight/
│                        hotel/tour/currency options) for the formal
│                        workload's destination catalog -- deliberately
│                        richer than development/'s 2-option-per-destination
│                        bundles (Step 6.5-7).
├── manifests/          -- generation/collection-plan manifests
│                        (workload manifest, formal_collection_plan.json,
│                        generation report).
└── splits/            -- primary_group_split.json / unseen_template_split.json
                          and their balance reports (Step 6.5-10..12).
```

## Generation discipline

All task instances and content bundles here are produced by a **deterministic
generator** (fixed seed + fixed `generator_version`) and then materialized to
JSON and committed -- never regenerated at random on each run. If the
generator or its inputs change, `generator_version` changes and prior
materialized instances are NOT silently overwritten in place.

## What's NOT here yet (Step 6.5A)

As of Step 6.5 Phase A, this directory holds only the specification
scaffolding (see `configs/travel_a2a/formal_workload/`) -- no task instances,
content bundles, or splits have been generated. That happens in Phase 6.5B.
