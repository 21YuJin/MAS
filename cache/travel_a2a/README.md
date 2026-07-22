# cache/travel_a2a

Raw Ollama telemetry / session cache for the `travel_a2a_v2` experiment
generation (A2A-inspired travel-booking multi-agent scenario).

```
experiment_version: travel_a2a_v2
environment_type:   a2a_inspired_travel
graph_source:        interaction_events
llm_backend:         ollama
model_name:          llama3.2
```

New top-level path -- the real_llm pilot keeps its cache inside
`output/real_llm/` (`cache_normal.json` / `cache_attack.json`), unchanged.
Nothing here overwrites or migrates that path.
