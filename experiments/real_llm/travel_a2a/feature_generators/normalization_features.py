"""
[Step 7B] Agent-normalized Feature Group -- SCAFFOLDING ONLY.

Per Step 7's explicit instruction, z-scores requiring normal-train
statistics are NOT computed in this phase: this module never fits
normal_statistics, never reads a training split, and never touches any raw
session data on its own. It only defines the deterministic z-score formula,
parameterized by an externally-supplied normal_statistics mapping. Fitting
that mapping is a later step's job -- it requires real per-agent
normal-session statistics, which do not exist until this workload is run
through ollama_runtime (still PLANNED_NOT_EXECUTED,
configs/travel_a2a/formal_workload/formal_collection_plan.json) and a
train-split-only statistics fit is performed (Step 8+, not here).

candidate_feature_registry.json (Step 7C) marks every feature built on top of
this module with requires_normal_statistics: true so no downstream consumer
can mistake this formula for a ready-to-use feature.
"""
from typing import Dict, Optional, Tuple

NormalStatsKey = Tuple[str, str]  # (agent_id, metric_name)


def agent_zscore(value: Optional[float], agent_id: str, metric_name: str,
                  normal_statistics: Dict[NormalStatsKey, Dict[str, float]]) -> Optional[float]:
    """(value - mean) / stdev using an externally-fit {(agent_id, metric_name): {"mean":..., "stdev":...}}
    mapping. Returns None if value is None, no stats exist for this
    (agent_id, metric_name), or stdev is 0 -- never raises, never guesses a
    default statistic."""
    if value is None:
        return None
    stats = normal_statistics.get((agent_id, metric_name))
    if stats is None or not stats.get("stdev"):
        return None
    return (value - stats["mean"]) / stats["stdev"]
