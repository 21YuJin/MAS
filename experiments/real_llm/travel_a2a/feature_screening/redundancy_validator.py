"""
[Phase 7D-D] Redundancy validator -- groups features via registry-declared
mathematically_dependent_on/potentially_redundant_with relationships
(union-find) and recommends ONE representative per group. Never removes a
feature: auto_remove is always False, and every original feature_name stays
in the registry untouched.
"""
from typing import Any, Dict, List


def _find(parent: Dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: Dict[str, str], a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def build_redundancy_groups(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    names = [f["feature_name"] for f in registry["features"]]
    parent = {n: n for n in names}
    relationship: Dict[frozenset, str] = {}

    for entry in registry["features"]:
        for ref in entry["mathematically_dependent_on"]:
            _union(parent, entry["feature_name"], ref)
            relationship[frozenset((entry["feature_name"], ref))] = "mathematically_dependent_on"
        for ref in entry["potentially_redundant_with"]:
            _union(parent, entry["feature_name"], ref)
            relationship.setdefault(frozenset((entry["feature_name"], ref)), "potentially_redundant_with")

    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(_find(parent, n), []).append(n)

    by_name = {f["feature_name"]: f for f in registry["features"]}

    def _dep_count(name: str) -> int:
        return len(by_name[name]["mathematically_dependent_on"])

    result: List[Dict[str, Any]] = []
    group_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_id += 1
        members_sorted = sorted(members)
        representative = sorted(members, key=lambda n: (_dep_count(n), n))[0]
        rel_types = sorted({
            relationship[frozenset((a, b))]
            for i, a in enumerate(members_sorted) for b in members_sorted[i + 1:]
            if frozenset((a, b)) in relationship
        })
        result.append({
            "redundancy_group_id": f"rg_{group_id:03d}",
            "members": members_sorted,
            "relationship_type": rel_types or ["unspecified"],
            "recommended_representative": representative,
            "auto_remove": False,
        })
    return result
