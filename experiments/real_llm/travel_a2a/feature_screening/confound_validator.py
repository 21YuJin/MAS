"""
[Phase 7D-F] Confound risk validator -- surfaces registry-declared
known_confound entries (CONFIRMED, from Step 6.5D's actual mock full-run
findings) plus a small set of WARNING-only additional watch candidates that
are structurally related to the confirmed ones. This validator never sets
candidate_only on a registry entry automatically -- a human decision stays
required before Step 8 acts on either list.
"""
from typing import Any, Dict, List

_CONFIRMED_CONFOUND_FEATURES = ("event_count", "message_count", "revision_count")

_ADDITIONAL_WATCH_FEATURES = (
    "session_features.call_count", "total_call_count", "artifact_count", "clarification_count",
    "unique_directed_pair_count", "graph_features.parallel_edge_count",
)


def confirmed_confound_report(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_name = {f["feature_name"]: f for f in registry["features"]}
    rows = []
    for name in _CONFIRMED_CONFOUND_FEATURES:
        entry = by_name.get(name)
        if entry is None or not entry["known_confound"]:
            continue
        rows.append({
            "feature_name": name,
            "known_confound": entry["known_confound"],
            "evidence_source": "reports/travel_a2a/formal_workload_dataset_card.md (Phase 6.5D mock full-run)",
            "severity": "confirmed",
            "recommended_control": "stratify by difficulty; do not adopt as a sole/core detection feature without controlling for it",
        })
    return rows


def additional_watch_report(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_name = {f["feature_name"]: f for f in registry["features"]}
    rows = []
    for name in _ADDITIONAL_WATCH_FEATURES:
        entry = by_name.get(name)
        if entry is None:
            continue
        rows.append({
            "feature_name": name,
            "known_confound": entry["known_confound"],
            "evidence_source": "heuristic -- structurally related to event_count/message_count (Phase 6.5D), not independently confirmed",
            "severity": "warning_only",
            "recommended_control": "empirically re-check for difficulty-tier separation once real ollama_runtime normal sessions exist, before Step 8 screening -- do not set candidate_only automatically",
        })
    return rows


def build_confound_risk_report(registry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "confirmed": confirmed_confound_report(registry),
        "additional_watch": additional_watch_report(registry),
        "note": "additional_watch entries are warnings only -- this validator never changes candidate_only on a registry entry automatically; that remains a human decision.",
    }
