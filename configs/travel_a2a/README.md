# configs/travel_a2a

Attack/task/topology config definitions for the `travel_a2a_v2` experiment
generation (A2A-inspired travel-booking multi-agent scenario).

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

Independent of `configs/attacks/` (real_llm pilot, unchanged -- see `v1` branch
and `experiments/real_llm/`). Nothing here overwrites or migrates that path.
