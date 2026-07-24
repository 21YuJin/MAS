"""
[Phase 7D-A] Registry integrity validator -- structural checks on
candidate_feature_registry.json only (no manifest, no raw schema, no
generator run needed). Never removes or mutates a feature entry; report only.
"""
from collections import Counter
from typing import Any, Dict, List

REQUIRED_KEYS = {
    "feature_name", "feature_level", "feature_family", "granularity", "source_fields", "formula", "unit", "dtype",
    "missing_value_policy", "normalization_policy", "requires_normal_statistics", "deployment_available",
    "provider_specific", "content_free", "candidate_only", "known_confound", "leakage_risk", "mock_availability",
    "ollama_required", "feature_role", "enabled", "derived_from_same_raw_group", "mathematically_dependent_on",
    "potentially_redundant_with",
}


def find_duplicate_feature_names(registry: Dict[str, Any]) -> List[str]:
    names = [f["feature_name"] for f in registry["features"]]
    return sorted(n for n, c in Counter(names).items() if c > 1)


def find_missing_required_keys(registry: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for entry in registry["features"]:
        missing = sorted(REQUIRED_KEYS - set(entry.keys()))
        if missing:
            out[entry["feature_name"]] = missing
    return out


def find_dangling_dependencies(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    names = {f["feature_name"] for f in registry["features"]}
    dangling = []
    for entry in registry["features"]:
        for ref in entry["mathematically_dependent_on"] + entry["potentially_redundant_with"]:
            if ref not in names:
                dangling.append({"feature_name": entry["feature_name"], "missing_reference": ref})
    return dangling


def find_cyclic_dependencies(registry: Dict[str, Any]) -> List[List[str]]:
    """DFS cycle detection over the mathematically_dependent_on graph only --
    a feature depending on a more primitive one. potentially_redundant_with is
    a symmetric hint, not a dependency direction, and is excluded here."""
    graph = {f["feature_name"]: f["mathematically_dependent_on"] for f in registry["features"]}
    visiting: set = set()
    visited: set = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]) -> None:
        if node in visiting:
            start = path.index(node)
            cycles.append(path[start:] + [node])
            return
        if node in visited or node not in graph:
            return
        visiting.add(node)
        for dep in graph[node]:
            dfs(dep, path + [node])
        visiting.discard(node)
        visited.add(node)

    for name in graph:
        if name not in visited:
            dfs(name, [])
    return cycles


def validate_registry_integrity(registry: Dict[str, Any]) -> Dict[str, Any]:
    duplicates = find_duplicate_feature_names(registry)
    missing_keys = find_missing_required_keys(registry)
    dangling = find_dangling_dependencies(registry)
    cycles = find_cyclic_dependencies(registry)
    return {
        "duplicate_feature_names": duplicates,
        "missing_required_keys": missing_keys,
        "dangling_dependencies": dangling,
        "cyclic_dependencies": cycles,
        "passed": not duplicates and not missing_keys and not dangling and not cycles,
    }
