"""
[Phase 7D-G] Deployment feasibility validator -- classifies each feature by
where it can actually be computed once outside this research harness:
portable / ollama_specific / runtime_specific / offline_only / diagnostic_only.
"""
from typing import Any, Dict, List

CATEGORIES = ("portable", "ollama_specific", "runtime_specific", "offline_only", "diagnostic_only")

_PORTABLE_FAMILIES = ("raw_session_aggregate", "graph_features", "session_dispersion")


def classify_deployment_feasibility(entry: Dict[str, Any]) -> str:
    if entry["feature_role"] == "collection_context":
        return "diagnostic_only"
    if entry["requires_normal_statistics"]:
        return "offline_only"
    if entry["provider_specific"]:
        return "ollama_specific"
    if entry["feature_family"] in _PORTABLE_FAMILIES:
        return "portable"
    return "runtime_specific"


def build_deployment_feasibility_report(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "feature_name": entry["feature_name"],
            "deployment_category": classify_deployment_feasibility(entry),
            "deployment_available": entry["deployment_available"],
        }
        for entry in registry["features"]
    ]
