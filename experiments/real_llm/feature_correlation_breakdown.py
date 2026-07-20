"""
Diagnose whether the pooled latency-token_count correlation (r=0.995 in
feature_ablation.py, computed over all normal sessions x all 4 agent roles
mixed together) is a genuine within-group relationship or an artifact of
pooling heterogeneous groups (different agent roles have very different
baseline output lengths; normal vs attack sessions differ in scale too).

Recomputes latency-token_count (and full 5x5) correlation:
  1. Normal sessions only, all roles pooled
  2. Attack sessions only, all roles pooled
  3. Per agent role, normal+attack pooled (role fixed, condition varies)
  4. Per agent role x condition (role fixed, condition fixed) -- smallest N (50)
     but the cleanest "within-group" estimate
"""

import json
import os
import numpy as np

TOPOLOGY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "topology_4agent_v1.json")


def load_topology(path):
    """Same validated loader as lgnn_experiment.py (see that file's docstring
    for the full check list). This script only needs `nodes`, but loads through
    the same validated path as the other headline scripts for consistency."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    nodes = cfg["nodes"]
    edges = [tuple(e) for e in cfg["edges"]]
    node_set = set(nodes)
    assert len(nodes) == len(node_set), f"duplicate node in topology: {nodes}"
    for a, b in edges:
        assert a in node_set and b in node_set, f"unknown node in edge: {(a, b)}"
    seen_edges = set()
    for a, b in edges:
        assert a != b, f"self-loop edge not allowed: {(a, b)}"
        key = frozenset((a, b))
        assert key not in seen_edges, f"duplicate edge: {(a, b)}"
        seen_edges.add(key)
    return {"topology_id": cfg["topology_id"], "nodes": nodes, "edges": edges}


_TOPOLOGY   = load_topology(TOPOLOGY_CONFIG_PATH)
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
# Generic IDs only, consistent with lgnn_experiment.py -- see AGENT_ROLES there for the
# example prompt roles actually used to collect these cached sessions.
AGENT_NAMES = _TOPOLOGY["nodes"]
AGENT_ROLES = {
    "Agent_0": "orchestration",
    "Agent_1": "research",
    "Agent_2": "analysis",
    "Agent_3": "writing",
}
N_FEATS     = len(FEAT_NAMES)

OUT = "./output/real_llm"
with open(os.path.join(OUT, "cache_normal.json")) as f:
    X_normal = np.array(json.load(f), dtype=np.float64)   # (50, 4, 5)
with open(os.path.join(OUT, "cache_attack.json")) as f:
    X_attack = np.array(json.load(f), dtype=np.float64)   # (50, 4, 5)

def lat_tok_r(X):
    """X: (N, 5) rows -> Pearson r between latency (col0) and token_count (col1)."""
    if len(X) < 3:
        return None
    return np.corrcoef(X[:, 0], X[:, 1])[0, 1]

def full_corr(X):
    return np.corrcoef(X, rowvar=False)

print("="*70)
print("  latency-token_count correlation breakdown")
print("="*70)

# 0. Baseline: everything pooled (normal+attack, all roles)
all_flat = np.concatenate([X_normal.reshape(-1, N_FEATS), X_attack.reshape(-1, N_FEATS)])
print(f"\n[0] 전체 pooled (정상+공격, 전체 role 혼합)  N={len(all_flat)}")
print(f"    r(latency, token_count) = {lat_tok_r(all_flat):.4f}")

# 1. Normal only, roles pooled
norm_flat = X_normal.reshape(-1, N_FEATS)
print(f"\n[1] 정상 세션만, role 혼합  N={len(norm_flat)}")
print(f"    r(latency, token_count) = {lat_tok_r(norm_flat):.4f}")

# 2. Attack only, roles pooled
atk_flat = X_attack.reshape(-1, N_FEATS)
print(f"\n[2] 공격 세션만, role 혼합  N={len(atk_flat)}")
print(f"    r(latency, token_count) = {lat_tok_r(atk_flat):.4f}")

# 3. Per agent role, normal+attack pooled (role fixed, condition varies)
print(f"\n[3] Agent role별 (정상+공격 pooled, role 고정)  N=100 each")
for i, name in enumerate(AGENT_NAMES):
    role_data = np.concatenate([X_normal[:, i, :], X_attack[:, i, :]])
    print(f"    {name:<14} r(latency, token_count) = {lat_tok_r(role_data):.4f}")

# 4. Per agent role x condition (cleanest within-group estimate, N=50 each)
print(f"\n[4] Agent role x condition (role+condition 둘 다 고정)  N=50 each")
print(f"    {'Agent':<14} {'Normal r':>10} {'Attack r':>10}")
for i, name in enumerate(AGENT_NAMES):
    rn = lat_tok_r(X_normal[:, i, :])
    ra = lat_tok_r(X_attack[:, i, :])
    print(f"    {name:<14} {rn:>10.4f} {ra:>10.4f}")

# Full 5x5 matrix for the cleanest within-group case (normal, per role) for reference
print("\n" + "="*70)
print("  참고: 정상 세션, role별 전체 5x5 상관행렬 대각선 밖 최대값")
print("="*70)
for i, name in enumerate(AGENT_NAMES):
    c = full_corr(X_normal[:, i, :])
    max_off = max(abs(c[a, b]) for a in range(N_FEATS) for b in range(N_FEATS) if a != b)
    print(f"    {name:<14} max|off-diag| = {max_off:.4f}")

print("\n완료.")
