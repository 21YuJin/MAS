"""
[Step 7B] Candidate Feature Generator package -- deterministic functions that
turn raw node/edge/session metadata (Phase 7A's raw_metadata_schema.json)
into candidate features. Raw records are read-only input here; nothing in
this package fits statistics, screens features, or trains anything.

Modules:
    token_features         -- node-level Ollama token counts (Phase 7A: node layer)
    timing_features        -- node-level Ollama durations + wall_clock_latency_ms
    graph_features          -- edge/artifact-lineage structural features
    session_features        -- within-session dispersion of node-level metrics
    normalization_features   -- z-score FORMULA only; never fits normal_statistics itself
    generate                -- per-session orchestrator combining the four raw-fed groups
    manifest                 -- feature_generation_manifest.json builder (diagnostic only)
"""
