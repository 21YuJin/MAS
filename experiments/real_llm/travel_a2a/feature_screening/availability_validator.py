"""
[Phase 7D-B] Availability validator -- classifies each registry feature by
what's actually needed to compute it TODAY, cross-checked against Phase 7B's
feature_generation_manifest.json null rates wherever a matching field exists.
Never computes a new null rate itself -- only reads the manifest Phase 7B
already produced.
"""
from typing import Any, Dict, List

AVAILABILITY_STATUSES = (
    "available_now_mock", "requires_ollama_runtime", "requires_normal_statistics", "planned_not_available",
)


def classify_feature_availability(entry: Dict[str, Any]) -> str:
    if entry["requires_normal_statistics"]:
        return "requires_normal_statistics"
    if entry["mock_availability"] == "requires_ollama_runtime":
        return "requires_ollama_runtime"
    if entry["mock_availability"] == "available_in_mock":
        return "available_now_mock"
    return "planned_not_available"


def _blocking_reason(status: str) -> Any:
    return {
        "available_now_mock": None,
        "requires_ollama_runtime": "requires an ollama_runtime session -- Ollama telemetry is never populated under deterministic_mock (mock_runner.py never sets llm_called=True).",
        "requires_normal_statistics": "requires a train-split-only fit of normal_statistics (Step 8+); this module never fits it itself.",
        "planned_not_available": "raw source field is planned_not_collected -- see raw_metadata_schema.json.",
    }[status]


def validate_feature_availability(registry: Dict[str, Any], generation_manifest: Dict[str, Any]) -> Dict[str, Any]:
    null_rates = generation_manifest.get("null_rate_by_field", {})
    rows: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    for entry in registry["features"]:
        status = classify_feature_availability(entry)
        rows.append({
            "feature_name": entry["feature_name"],
            "availability_status": status,
            "blocking_reason": _blocking_reason(status),
        })
        observed_null_rate = null_rates.get(entry["feature_name"])
        if observed_null_rate is None:
            continue
        if status == "requires_ollama_runtime" and observed_null_rate < 1.0:
            mismatches.append({
                "feature_name": entry["feature_name"],
                "reason": "registry claims requires_ollama_runtime but the manifest shows some non-null values under mock",
                "observed_null_rate": observed_null_rate,
            })
        if status == "available_now_mock" and observed_null_rate > 0.0:
            mismatches.append({
                "feature_name": entry["feature_name"],
                "reason": "registry claims available_now_mock but the manifest shows nulls under mock",
                "observed_null_rate": observed_null_rate,
            })
    return {"rows": rows, "manifest_mismatches": mismatches, "passed": not mismatches}
