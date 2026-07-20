"""
Strong-baseline + graph ablation on the real-LLM (Ollama llama3.2) cached
sessions (output/real_llm/cache_*.json) -- 16 step. No new Ollama calls: this
reuses the existing real_llm_v1 cache, exactly like feature_ablation.py.

Question: does LightGAE's advantage actually come from the graph structure
(the 4 real communication edges), or would any model that mixes information
across agents do just as well on this fixed topology?

Six methods, all sharing split/scaler/threshold-policy/feature-set/parameter
budget/optimizer/epoch/seed with the headline run (lgnn_experiment.py):

  1. Z-score            -- no learned model, per-session L2 norm of scaled Core-2 features
  2. Node-wise MLP-AE    -- shared-weight per-node AE, NO cross-node mixing at all
                            (the true "no graph, no flatten" floor)
  3. Flattened MLP-AE    -- single dense AE over all 4 agents concatenated
                            (mixes every agent with every other agent, but with
                            no topology prior -- the strongest non-graph competitor
                            on a fixed topology). This is what lgnn_experiment.py's
                            "MLPAE" ablation already is (X.reshape(B,-1) before
                            encoding) -- reused here under its accurate name.
  4. LightGAE No-edge    -- GCN with adjacency = identity (self-loop only).
                            Mathematically reduces to a per-node Linear layer with
                            shared weights, i.e. architecturally the same computation
                            as Node-wise MLP-AE built through GCNLayer instead of a
                            plain nn.Linear encoder/decoder. If method 2 and 4 land
                            close, that's an internal consistency check, not a bug.
  5. LightGAE Random-edge -- GCN with |E|=4 edges resampled uniformly (no self-loop
                            duplicate) at each seed, same count as the real topology.
  6. LightGAE Correct-edge (= headline LightGAE) -- the real topology_4agent_v1 edges.

Interpretation (printed at the end, also in baseline_ablation_results.json):
  Correct >> Random/No-edge  -> the real communication edges carry information
  Correct ~= Random ~= No-edge -> edge choice contributes little under this
                                   fixed topology / attack design
  Flattened ~= Correct        -> no explicit graph-structure advantage over a
                                   plain dense mixer on this fixed topology
Whatever the outcome is, it bounds the paper's graph-structure claim -- this
script's job is to report the honest comparison, not to make LightGAE win.
"""
import json
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

TOPOLOGY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "topology_4agent_v1.json")


def load_topology(path):
    """Same validated loader as lgnn_experiment.py -- duplicated (not imported)
    so this script keeps running standalone, consistent with this codebase's style."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    nodes = cfg["nodes"]
    edges = [tuple(e) for e in cfg["edges"]]
    primary_predecessor = cfg["primary_predecessor"]
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
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    visited, frontier = {nodes[0]}, [nodes[0]]
    while frontier:
        cur = frontier.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt); frontier.append(nxt)
    assert not (node_set - visited), f"disconnected node(s) in topology: {node_set - visited}"
    assert set(primary_predecessor.keys()) == node_set
    for node, pred in primary_predecessor.items():
        if pred is None:
            continue
        assert pred in node_set
        assert frozenset((node, pred)) in seen_edges
    assert any(p is None for p in primary_predecessor.values())
    return {"topology_id": cfg["topology_id"], "nodes": nodes, "edges": edges,
            "primary_predecessor": primary_predecessor}


_TOPOLOGY   = load_topology(TOPOLOGY_CONFIG_PATH)
TOPOLOGY_ID = _TOPOLOGY["topology_id"]
AGENT_NAMES = _TOPOLOGY["nodes"]
N_AGENTS    = len(AGENT_NAMES)
EDGES       = [(AGENT_NAMES.index(a), AGENT_NAMES.index(b)) for a, b in _TOPOLOGY["edges"]]
N_EDGES     = len(EDGES)
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
N_FEATS     = len(FEAT_NAMES)
CORE_FEATURES = ["token_count", "ctx_delta"]   # headline feature set (see lgnn_experiment.py)
CORE_COLS     = [FEAT_NAMES.index(f) for f in CORE_FEATURES]
N_CORE        = len(CORE_COLS)


def build_adj(edges, n_agents=N_AGENTS):
    A = np.zeros((n_agents, n_agents), dtype=np.float32)
    for s, d in edges:
        A[s, d] = A[d, s] = 1.0
    A += np.eye(n_agents, dtype=np.float32)
    deg  = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)


def random_edges(seed, n_agents=N_AGENTS, n_edges=N_EDGES):
    """|E| = n_edges pairs sampled uniformly without replacement from all
    possible node pairs -- a fresh draw per seed (own RNG stream, offset well
    clear of the model-init/split seed range, so it doesn't correlate with
    which sessions land in train/val/test for that seed)."""
    rng = np.random.RandomState(20000 + seed)
    all_pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    chosen = rng.choice(len(all_pairs), size=n_edges, replace=False)
    return [all_pairs[k] for k in chosen]


ADJ_CORRECT = build_adj(EDGES)
ADJ_NOEDGE  = build_adj([])   # self-loop only

OUT = "./output/real_llm"
CACHE_NORMAL = os.path.join(OUT, "cache_normal.json")
CACHE_ATTACK = os.path.join(OUT, "cache_attack.json")

with open(CACHE_NORMAL) as f:
    X_normal_full = np.array(json.load(f), dtype=np.float32)   # (50, 4, 5)
with open(CACHE_ATTACK) as f:
    X_attack_full = np.array(json.load(f), dtype=np.float32)   # (50, 4, 5)

N_NORMAL = len(X_normal_full)
N_ATTACK = len(X_attack_full)
print(f"Loaded cache: normal={N_NORMAL}  attack={N_ATTACK}  (no Ollama calls -- cache only)")

N_TASKS = 20  # must match len(TASKS) in lgnn_experiment.py
task_id_normal = np.array([i % N_TASKS for i in range(N_NORMAL)])

THRESHOLD_PERCENTILE   = 95
SEEDS                  = [42, 0, 1, 7, 123]
NORMAL_SPLIT_FRACTIONS = {"train": 0.60, "val": 0.20, "test": 0.20}
N_TR        = int(round(N_NORMAL * NORMAL_SPLIT_FRACTIONS["train"]))
N_VAL       = int(round(N_NORMAL * NORMAL_SPLIT_FRACTIONS["val"]))
N_TE_NORMAL = N_NORMAL - N_TR - N_VAL
EPOCHS, LR, BS, WD = 160, 1e-3, 16, 1e-4


# ══════════════════════════════════════════════════════════════
# §1. Models
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H, A):
        return self.linear(torch.einsum("ij,bjk->bik", A, H))


class LightGAE(nn.Module):
    """Same architecture as lgnn_experiment.py's headline model; adjacency A is
    passed in at call time so one class serves all 3 graph-ablation conditions."""
    def __init__(self, in_dim, hid=16, emb=8):
        super().__init__()
        self.gc1  = GCNLayer(in_dim, hid)
        self.gc2  = GCNLayer(hid, emb)
        self.dec1 = nn.Linear(emb, hid)
        self.dec2 = nn.Linear(hid, in_dim)

    def forward(self, X, A):
        H1 = F.relu(self.gc1(X, A))
        H1 = F.dropout(H1, p=0.1, training=self.training)
        H2 = self.gc2(H1, A)
        return self.dec2(F.relu(self.dec1(H2)))

    @torch.no_grad()
    def score(self, X_t, A):
        self.eval()
        X_hat = self.forward(X_t, A)
        return ((X_t - X_hat) ** 2).mean(dim=2).mean(dim=1).numpy()


class NodeWiseMLPAE(nn.Module):
    """Shared-weight per-node autoencoder -- NO cross-node mixing (nn.Linear
    broadcasts over the agent dimension, so every node is encoded/decoded by
    the exact same weights, independently of every other node's features)."""
    def __init__(self, in_dim, hid=16, emb=8):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hid)
        self.enc2 = nn.Linear(hid, emb)
        self.dec1 = nn.Linear(emb, hid)
        self.dec2 = nn.Linear(hid, in_dim)

    def forward(self, X):
        H1 = F.relu(self.enc1(X))
        H1 = F.dropout(H1, p=0.1, training=self.training)
        H2 = self.enc2(H1)
        return self.dec2(F.relu(self.dec1(H2)))

    @torch.no_grad()
    def score(self, X_t):
        self.eval()
        X_hat = self.forward(X_t)
        return ((X_t - X_hat) ** 2).mean(dim=2).mean(dim=1).numpy()


class FlattenedMLPAE(nn.Module):
    """All 4 agents concatenated into one vector before encoding -- mixes every
    agent with every other agent through dense weights, but with no topology
    prior at all. This is lgnn_experiment.py's "MLPAE" ablation, renamed here
    to match what it actually computes."""
    def __init__(self, n_agents, n_feats, hid=16, emb=8):
        super().__init__()
        self.n_agents, self.n_feats = n_agents, n_feats
        self.enc = nn.Sequential(
            nn.Linear(n_agents * n_feats, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, emb))
        self.dec = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, n_agents * n_feats))

    def forward(self, X):
        B = X.shape[0]
        z = self.enc(X.reshape(B, -1))
        return self.dec(z).reshape(B, self.n_agents, self.n_feats)

    @torch.no_grad()
    def score(self, X_t):
        self.eval()
        X_hat = self.forward(X_t)
        return ((X_t - X_hat) ** 2).mean(dim=2).mean(dim=1).numpy()


def train_ae(model, X_normal, epochs=EPOCHS, lr=LR, bs=BS, A=None):
    """One training loop for all 4 learned methods (Node-wise/Flattened MLP-AE
    take no A; the 3 LightGAE variants pass their adjacency in)."""
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    for _ in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i + bs]]
            X_hat = model(b, A) if A is not None else model(b)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def metrics(y, sc, pred):
    if len(np.unique(y)) < 2:
        return dict(TPR=0.0, FPR=0.0, precision=0.0, F1=0.0, AUC=0.5)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        TPR=round(tp / (tp + fn + 1e-8), 4),
        FPR=round(fp / (fp + tn + 1e-8), 4),
        precision=round(tp / (tp + fp + 1e-8), 4),
        F1=round(f1_score(y, pred, zero_division=0), 4),
        AUC=round(roc_auc_score(y, sc), 4),
    )


def group_split_3way(group_ids, seed, n_train, n_val, n_test):
    """Same policy as lgnn_experiment.py's group_split_3way -- see that file
    for the full docstring."""
    group_ids = np.asarray(group_ids)
    members = {}
    for i, g in enumerate(group_ids):
        members.setdefault(int(g), []).append(i)
    order = list(members.keys())
    np.random.RandomState(seed).shuffle(order)
    targets = {"train": n_train, "val": n_val, "test": n_test}
    counts  = {"train": 0, "val": 0, "test": 0}
    bucket  = {"train": [], "val": [], "test": []}
    for g in order:
        idxs = members[g]
        deficit = {k: targets[k] - counts[k] for k in targets}
        dest = max(deficit, key=deficit.get)
        bucket[dest].extend(idxs)
        counts[dest] += len(idxs)
    return (np.array(sorted(bucket["train"])), np.array(sorted(bucket["val"])),
            np.array(sorted(bucket["test"])))


# ══════════════════════════════════════════════════════════════
# §2. Ablation driver
# ══════════════════════════════════════════════════════════════

METHOD_NAMES = ["Z-score", "Node-wise MLP-AE", "Flattened MLP-AE",
                 "LightGAE No-edge", "LightGAE Random-edge", "LightGAE Correct-edge"]
results = {name: {"AUC": [], "F1": [], "precision": [], "TPR": [], "FPR": []} for name in METHOD_NAMES}

print("=" * 78)
print("  Strong Baseline + Graph Ablation -- Real-LLM 4-Agent MAS (Ollama llama3.2)")
print("=" * 78)
print("\n  Learning setup: Normal-only novelty detection (identical to headline run)")
print(f"    Normal train:      {N_TR:3d}   Normal validation: {N_VAL:3d}   "
      f"Normal test: {N_TE_NORMAL:3d}   Attack test: {N_ATTACK:3d}")
print(f"    Feature set: Core-2 {CORE_FEATURES}   Threshold: percentile({THRESHOLD_PERCENTILE}, normal-val)")
print(f"    Shared across all 6 methods: split(seed), scaler, threshold policy, "
      f"epochs={EPOCHS}, lr={LR}, bs={BS}, optimizer=Adam(wd={WD}), hid=16, emb=8")

for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)

    idx_tr, idx_val, idx_ten = group_split_3way(task_id_normal, s, N_TR, N_VAL, N_TE_NORMAL)
    assert len(idx_tr) + len(idx_val) + len(idx_ten) == N_NORMAL
    tids_tr, tids_val, tids_ten = (set(task_id_normal[idx_tr].tolist()),
                                    set(task_id_normal[idx_val].tolist()),
                                    set(task_id_normal[idx_ten].tolist()))
    assert not (tids_tr & tids_val) and not (tids_tr & tids_ten) and not (tids_val & tids_ten), \
        "group split leaked a task_id across train/val/test"

    X_tr_raw, X_val_raw, X_ten_raw = X_normal_full[idx_tr], X_normal_full[idx_val], X_normal_full[idx_ten]
    scaler = StandardScaler().fit(X_tr_raw.reshape(len(X_tr_raw), -1))
    assert scaler.n_samples_seen_ == len(idx_tr), "scaler must be fit on training-normal sessions only"

    def _scale(X_raw):
        return scaler.transform(X_raw.reshape(len(X_raw), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)

    X_tr_all, X_val_all, X_ten_all = _scale(X_tr_raw), _scale(X_val_raw), _scale(X_ten_raw)
    Xa_all   = _scale(X_attack_full)
    X_te_all = np.concatenate([X_ten_all, Xa_all])
    y_te     = np.array([0] * len(X_ten_all) + [1] * N_ATTACK)

    X_tr, X_val, X_te = X_tr_all[:, :, CORE_COLS], X_val_all[:, :, CORE_COLS], X_te_all[:, :, CORE_COLS]
    assert X_tr.shape[0] == len(idx_tr), "model input (X_tr) must be normal-train-only"

    print(f"\n  seed={s}  split(train/val/test_normal)={len(idx_tr)}/{len(idx_val)}/{len(idx_ten)}")

    # ── Z-score (no learned model) ──────────────────────────────
    flat_tr, flat_val, flat_te = X_tr.reshape(len(X_tr), -1), X_val.reshape(len(X_val), -1), X_te.reshape(len(X_te), -1)
    zsc  = StandardScaler().fit(flat_tr)
    zval = np.linalg.norm(zsc.transform(flat_val), axis=1)
    zte  = np.linalg.norm(zsc.transform(flat_te), axis=1)
    z_th = float(np.percentile(zval, THRESHOLD_PERCENTILE))
    m = metrics(y_te, zte, (zte > z_th).astype(int))
    for k in results["Z-score"]:
        results["Z-score"][k].append(m[k])
    print(f"    {'Z-score':<24} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

    # ── Node-wise MLP-AE ─────────────────────────────────────────
    nw = NodeWiseMLPAE(in_dim=N_CORE, hid=16, emb=8)
    train_ae(nw, X_tr)
    sc_val, sc_te = nw.score(torch.FloatTensor(X_val)), nw.score(torch.FloatTensor(X_te))
    th = float(np.percentile(sc_val, THRESHOLD_PERCENTILE))
    m = metrics(y_te, sc_te, (sc_te > th).astype(int))
    for k in results["Node-wise MLP-AE"]:
        results["Node-wise MLP-AE"][k].append(m[k])
    print(f"    {'Node-wise MLP-AE':<24} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

    # ── Flattened MLP-AE ─────────────────────────────────────────
    fl = FlattenedMLPAE(n_agents=N_AGENTS, n_feats=N_CORE, hid=16, emb=8)
    train_ae(fl, X_tr)
    sc_val, sc_te = fl.score(torch.FloatTensor(X_val)), fl.score(torch.FloatTensor(X_te))
    th = float(np.percentile(sc_val, THRESHOLD_PERCENTILE))
    m = metrics(y_te, sc_te, (sc_te > th).astype(int))
    for k in results["Flattened MLP-AE"]:
        results["Flattened MLP-AE"][k].append(m[k])
    print(f"    {'Flattened MLP-AE':<24} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

    # ── LightGAE: No-edge / Random-edge / Correct-edge ────────────
    adj_variants = {
        "LightGAE No-edge": ADJ_NOEDGE,
        "LightGAE Random-edge": build_adj(random_edges(s)),
        "LightGAE Correct-edge": ADJ_CORRECT,
    }
    for name, A in adj_variants.items():
        gae = LightGAE(in_dim=N_CORE, hid=16, emb=8)
        train_ae(gae, X_tr, A=A)
        sc_val = gae.score(torch.FloatTensor(X_val), A)
        sc_te  = gae.score(torch.FloatTensor(X_te), A)
        th = float(np.percentile(sc_val, THRESHOLD_PERCENTILE))
        m = metrics(y_te, sc_te, (sc_te > th).astype(int))
        for k in results[name]:
            results[name][k].append(m[k])
        print(f"    {name:<24} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

# ══════════════════════════════════════════════════════════════
# §3. Summary + interpretation
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("  Multi-seed summary (N=5 seeds)")
print("=" * 78)
print(f"  {'Method':<24} {'AUC':>17} {'F1':>17} {'Precision':>12} {'Recall':>10} {'FPR':>8}")
print("  " + "-" * 92)
summary = {}
for name in METHOD_NAMES:
    r = results[name]
    auc, f1 = np.array(r["AUC"]), np.array(r["F1"])
    summary[name] = {
        "auc_mean": float(auc.mean()), "auc_std": float(auc.std()),
        "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
        "precision_mean": float(np.mean(r["precision"])),
        "recall_mean": float(np.mean(r["TPR"])),
        "fpr_mean": float(np.mean(r["FPR"])),
    }
    print(f"  {name:<24} {auc.mean():.4f}±{auc.std():.4f}  {f1.mean():.4f}±{f1.std():.4f}  "
          f"{np.mean(r['precision']):>10.4f}  {np.mean(r['TPR']):>8.4f}  {np.mean(r['FPR']):>6.4f}")

t_cr, p_cr = stats.ttest_rel(results["LightGAE Correct-edge"]["F1"], results["LightGAE Random-edge"]["F1"])
t_cn, p_cn = stats.ttest_rel(results["LightGAE Correct-edge"]["F1"], results["LightGAE No-edge"]["F1"])
t_cf, p_cf = stats.ttest_rel(results["LightGAE Correct-edge"]["F1"], results["Flattened MLP-AE"]["F1"])
print(f"\n  [paired t-test, F1, N=5 seeds]")
print(f"  Correct-edge vs Random-edge : t={t_cr:+.3f}  p={p_cr:.4f}")
print(f"  Correct-edge vs No-edge     : t={t_cn:+.3f}  p={p_cn:.4f}")
print(f"  Correct-edge vs Flattened   : t={t_cf:+.3f}  p={p_cf:.4f}")

d_random = summary["LightGAE Correct-edge"]["auc_mean"] - summary["LightGAE Random-edge"]["auc_mean"]
d_noedge = summary["LightGAE Correct-edge"]["auc_mean"] - summary["LightGAE No-edge"]["auc_mean"]
d_flat   = summary["LightGAE Correct-edge"]["auc_mean"] - summary["Flattened MLP-AE"]["auc_mean"]
d_nodewise_noedge = abs(summary["Node-wise MLP-AE"]["auc_mean"] - summary["LightGAE No-edge"]["auc_mean"])

print("\n  interpretation:")
print(f"    ΔAUC(Correct-Random) = {d_random:+.4f}, ΔAUC(Correct-NoEdge) = {d_noedge:+.4f}  "
      f"-> {'real edges carry signal' if (d_random > 0.01 or d_noedge > 0.01) else 'edge choice contributes little under this fixed topology/attack design'}")
print(f"    ΔAUC(Correct-Flattened) = {d_flat:+.4f}  "
      f"-> {'graph prior beats a plain dense mixer' if d_flat > 0.01 else 'no explicit graph-structure advantage over a flat dense mixer on this fixed topology'}")
print(f"    |Node-wise MLP-AE - LightGAE No-edge| AUC = {d_nodewise_noedge:.4f}  "
      f"(sanity check: these two should be architecturally near-equivalent -- "
      f"{'consistent' if d_nodewise_noedge < 0.02 else 'diverged more than expected, worth investigating'})")

OUT_PATH = os.path.join(OUT, "baseline_ablation_results.json")
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "purpose": "16 step -- strong baseline + graph ablation, reusing the existing "
                   "real_llm_v1 cache (no new Ollama calls)",
        "shared_config": {
            "seeds": SEEDS, "threshold_percentile": THRESHOLD_PERCENTILE,
            "core_features": CORE_FEATURES, "epochs": EPOCHS, "lr": LR, "batch_size": BS,
            "weight_decay": WD, "hid": 16, "emb": 8,
            "split": {"normal_train": N_TR, "normal_val": N_VAL,
                      "normal_test": N_TE_NORMAL, "attack_test": N_ATTACK},
        },
        "methods": summary,
        "significance_tests": {
            "correct_vs_random_edge": {"t": float(t_cr), "p": float(p_cr)},
            "correct_vs_no_edge": {"t": float(t_cn), "p": float(p_cn)},
            "correct_vs_flattened_mlpae": {"t": float(t_cf), "p": float(p_cf)},
        },
        "interpretation": {
            "auc_delta_correct_minus_random": d_random,
            "auc_delta_correct_minus_noedge": d_noedge,
            "auc_delta_correct_minus_flattened": d_flat,
            "nodewise_vs_noedge_auc_gap": d_nodewise_noedge,
        },
    }, f, indent=2)
print(f"\n  [saved] {OUT_PATH}")
print("\n실험 완료.")
