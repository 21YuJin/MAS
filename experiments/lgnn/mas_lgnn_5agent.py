"""
5-Agent Extended LightGAE: Demonstrating GCN Structural Advantage
in Complex Multi-Agent AI System Security

Key Contributions:
  - Extended MAS model G5=(A5,E8,M): 5 agents, 8 communication edges
  - Type-V Chain: 3-hop cascading attack (Planner→Researcher→Analyst→Writer)
  - GCN structural inductive bias advantage over flat MLP-AE, especially on
    low-signal chain propagation attacks (Type-III Slow, Type-V Chain)
  - Each downstream agent's features causally depend on upstream context,
    creating 3-hop inter-agent correlations that GCN's message passing exploits

System Model G5:
  A5 = {Orchestrator(v0), Planner(v1), Researcher(v2), Analyst(v3), Writer(v4)}
  E8 = pipeline:    (v0→v1, v1→v2, v2→v3, v3→v4)
       supervisory: (v0→v2, v0→v3, v0→v4)
       cross-link:  (v1→v3)
  M  = {δ:latency, τ:token_count, Δc:ctx_delta, f:sentence_count, s:joint_deviation_flag}
"""

import time
import warnings
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

BLUE   = "#4C9BE8"
RED    = "#E8604C"
GREEN  = "#5BAD6F"
ORANGE = "#F0A500"
PURPLE = "#9B59B6"
GRAY   = "#AAAAAA"
TEAL   = "#1ABC9C"

OUT = "./output/lgnn_5agent"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# §1.  5-AGENT MAS GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════

N_AGENTS    = 5
AGENT_NAMES = ["Orchestrator", "Planner", "Researcher", "Analyst", "Writer"]
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
N_FEATS     = len(FEAT_NAMES)

# Headline LightGAE model uses the empirically-selected Core-2 subset (see
# experiments/lgnn/feature_ablation_5agent.py): latency was dropped after
# ablation showed it added no value on real-LLM data (near-perfectly
# redundant with token_count there) and only a small gain in this synthetic
# simulation. All 5 raw features are still collected/plotted for feature
# distribution stats; only CORE_COLS are fed to the model.
CORE_COLS   = [1, 2]   # token_count, ctx_delta
CORE_NAMES  = [FEAT_NAMES[i] for i in CORE_COLS]
N_CORE      = len(CORE_COLS)

#  Pipeline:    0→1→2→3→4
#  Supervisory: 0→2, 0→3, 0→4
#  Cross-link:  1→3  (Planner coordinates Analyst directly)
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,2),(0,3),(0,4),(1,3)]

def build_adj() -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops: D^{-1/2}(A+I)D^{-1/2}"""
    A = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float32)
    for s, d in EDGES:
        A[s,d] = A[d,s] = 1.0
    A += np.eye(N_AGENTS, dtype=np.float32)
    deg  = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)

ADJ = build_adj()

# ══════════════════════════════════════════════════════════════
# §2.  DATA GENERATION -- 4 ORIGINAL + NEW TYPE-V CHAIN
# ══════════════════════════════════════════════════════════════

NP = dict(latency=(0.85, 0.12), token_count=(160, 25),
          sentence_count=2.5,    ctx_delta=(0.05, 0.02))
AP = dict(latency=(1.30, 0.30), token_count=(240, 50),
          sentence_count=5.5,    ctx_delta=(0.18, 0.06))

ATTACK_CFG = {
    "Normal":          {"p_pln":0.00, "p_res":0.00, "p_ana":0.00, "p_wrt":0.00},
    "Type-I Direct":   {"p_pln":0.00, "p_res":1.00, "p_ana":0.00, "p_wrt":0.00},
    "Type-II Harvest": {"p_pln":0.00, "p_res":0.80, "p_ana":0.35, "p_wrt":0.00},
    "Type-III Slow":   {"p_pln":0.00, "p_res":0.30, "p_ana":0.10, "p_wrt":0.05},
    "Type-IV Flood":   {"p_pln":0.00, "p_res":0.65, "p_ana":0.50, "p_wrt":0.65},
    "Type-V Chain":    {"p_pln":0.40, "p_res":0.00, "p_ana":0.00, "p_wrt":0.00},
}
COLORS_ATK = {
    "Normal":          BLUE,
    "Type-I Direct":   RED,
    "Type-II Harvest": ORANGE,
    "Type-III Slow":   PURPLE,
    "Type-IV Flood":   GREEN,
    "Type-V Chain":    TEAL,
}

def _lerp(n_val, a_val, p):
    if isinstance(n_val, tuple):
        return n_val[0]+p*(a_val[0]-n_val[0]), n_val[1]+p*(a_val[1]-n_val[1])
    return n_val + p*(a_val - n_val)

def sample_agent(p: float, context_scale: float = 1.0) -> list:
    """
    Stronger coupling than 3-agent version: upstream context propagates more
    forcefully to model the 3-hop pipeline dependency.
      lat_scale ∈ [0.5, ~1.35],  ctx_scale ∈ [0.3, ~1.55]
    This amplifies inter-agent feature correlation that GCN can exploit.
    """
    mu_l, sg_l = _lerp(NP["latency"],     AP["latency"],     p)
    mu_t, sg_t = _lerp(NP["token_count"], AP["token_count"], p)
    mu_c, sg_c = _lerp(NP["ctx_delta"],   AP["ctx_delta"],   p)
    lam_a      = _lerp(NP["sentence_count"], AP["sentence_count"], p)
    lat_scale  = 0.5 + 0.5 * context_scale
    ctx_scale  = 0.3 + 0.7 * context_scale
    lat_val = max(0.05, np.random.normal(mu_l * lat_scale, sg_l))
    tok_val = max(10,   int(np.random.normal(mu_t, sg_t)))
    # joint_deviation_flag: joint latency+token deviation flag, derived from the *realized*
    # lat/tok values (not sampled directly from p) to avoid label leakage.
    lat_z = (lat_val - NP["latency"][0]) / NP["latency"][1]
    tok_z = (tok_val - NP["token_count"][0]) / NP["token_count"][1]
    return [
        lat_val,
        tok_val,
        max(0.0,  np.random.normal(mu_c * ctx_scale, sg_c)),
        max(0,    int(np.random.poisson(lam_a))),
        int(lat_z > 1.5 and tok_z > 1.0),
    ]

def make_session(atk_key="Normal", n_turns=30, win=5):
    """
    3-hop causal chain: Orchestrator -> Planner -> Researcher -> Analyst -> Writer

    Context propagation uses combined signal: max(token_ratio, latency_ratio).
    This enables latency cascade even when downstream tokens stay normal (p=0),
    modelling how pipeline latency anomalies propagate without explicit compromise.

    Type-V Chain: single injection at Planner only (p_pln=0.40). Researcher,
    Analyst, and Writer are NOT directly compromised -- only their latency is
    elevated via upstream_signal cascade. Individual agents appear borderline
    normal; the correlated latency pattern across connected nodes is the signal
    that GCN's message passing can exploit while flat MLP-AE cannot.
    """
    cfg     = ATTACK_CFG[atk_key]
    label   = 0 if atk_key == "Normal" else 1
    tok_rng = AP["token_count"][0] - NP["token_count"][0]   # 80

    def eff_p(p_base, upstream_tokens):
        if p_base <= 0:
            return 0.0
        excess = max(0.0, upstream_tokens - NP["token_count"][0]) / (tok_rng + 1e-8)
        return p_base * min(1.0, 0.2 + 0.8 * excess)

    def upstream_signal(feats):
        """Combined upstream signal: max(token_ratio, latency_ratio).
        Latency propagates the cascade even when downstream tokens are normal,
        enabling multi-hop detection without direct per-node compromise.
        """
        return max(feats[1] / NP["token_count"][0],
                   feats[0] / NP["latency"][0])

    turns = []
    for _ in range(n_turns):
        orch = sample_agent(0.0)

        pln  = sample_agent(cfg["p_pln"])
        pln_sig = upstream_signal(pln)

        p_res = eff_p(cfg["p_res"], pln[1]) if cfg["p_pln"] > 0 else cfg["p_res"]
        res   = sample_agent(p_res, context_scale=pln_sig)
        res_sig = upstream_signal(res)

        ana   = sample_agent(eff_p(cfg["p_ana"], res[1]), context_scale=res_sig)
        ana_sig = upstream_signal(ana)

        wrt   = sample_agent(eff_p(cfg["p_wrt"], ana[1]), context_scale=ana_sig)

        turns.append([orch, pln, res, ana, wrt])

    out = []
    for i in range(win, n_turns + 1):
        X = np.mean(turns[i-win:i], axis=0).astype(np.float32)
        out.append((X, label, atk_key))
    return out

def build_dataset(n_sess=60, n_turns=30, win=5):
    Xs, ys, ts = [], [], []
    for ak in ATTACK_CFG:
        for _ in range(n_sess):
            for X, y, t in make_session(ak, n_turns, win):
                Xs.append(X); ys.append(y); ts.append(t)
    return np.array(Xs), np.array(ys), np.array(ts)

# ══════════════════════════════════════════════════════════════
# §3.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """H' = σ(Â H W),  Â = normalized adjacency with self-loops."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H, A):
        return self.linear(torch.einsum("ij,bjk->bik", A, H))


class LightGAE(nn.Module):
    """
    Lightweight Graph Autoencoder for one-class anomaly detection.
    2-layer GCN encoder captures 2-hop neighborhood (v_i sees v_j and v_k
    when j∈N(i) and k∈N(j)), enabling 3-hop chain correlation detection.
    """
    def __init__(self, in_dim=N_FEATS, hid=16, emb=8, verbose=True):
        super().__init__()
        self.gc1  = GCNLayer(in_dim, hid)
        self.gc2  = GCNLayer(hid,    emb)
        self.dec1 = nn.Linear(emb, hid)
        self.dec2 = nn.Linear(hid, in_dim)
        if verbose:
            n = sum(p.numel() for p in self.parameters())
            print(f"  LightGAE  hid={hid}  emb={emb}  params={n:,}")

    def encode(self, X, A):
        H1 = F.relu(self.gc1(X, A))
        H1 = F.dropout(H1, p=0.1, training=self.training)
        H2 = self.gc2(H1, A)
        return H2, H2.mean(dim=1)

    def decode(self, H2):
        return self.dec2(F.relu(self.dec1(H2)))

    def forward(self, X, A):
        H2, z = self.encode(X, A)
        return self.decode(H2), H2, z

    @torch.no_grad()
    def score(self, X_t, A):
        self.eval()
        X_hat, H2, z = self.forward(X_t, A)
        node_err  = ((X_t - X_hat) ** 2).mean(dim=2)
        graph_err = node_err.mean(dim=1)
        return graph_err.numpy(), node_err.numpy(), z.numpy()


class MLPAE(nn.Module):
    """
    Flat MLP Autoencoder -- concatenates all agent features, no message passing.
    Cannot exploit graph structure; treats agent relationships as implicit correlations
    in a flat 25-dim input. Ablation baseline for GCN structural benefit.
    """
    def __init__(self, in_dim=N_AGENTS * N_FEATS, n_feats=N_FEATS, hid=16, emb=8, verbose=True):
        super().__init__()
        self.n_feats = n_feats
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid, emb))
        self.dec = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Linear(hid, in_dim))
        if verbose:
            n = sum(p.numel() for p in self.parameters())
            print(f"  MLPAE     hid={hid}  emb={emb}  params={n:,}")

    def forward(self, X):
        B = X.shape[0]
        return self.dec(self.enc(X.reshape(B, -1))).reshape(B, N_AGENTS, self.n_feats)

    @torch.no_grad()
    def score(self, X_t):
        self.eval()
        return ((X_t - self.forward(X_t)) ** 2).mean(dim=2).mean(dim=1).numpy()


def train_gae(model, X_normal, epochs=150, lr=1e-3, bs=64, verbose=True):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        ep_loss = 0.0
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i+bs]]
            X_hat, _, _ = model(b, ADJ)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  loss={ep_loss:.5f}")


def train_mlp(model, X_normal, epochs=150, lr=1e-3, bs=64):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b    = X_t[idx[i:i+bs]]
            loss = F.mse_loss(model(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def metrics(y, sc, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(
        TPR=round(tp/(tp+fn+1e-8), 4),
        FPR=round(fp/(fp+tn+1e-8), 4),
        F1 =round(f1_score(y, pred, zero_division=0), 4),
        AUC=round(roc_auc_score(y, sc), 4),
    )

# ══════════════════════════════════════════════════════════════
# §4.  EXPERIMENT
# ══════════════════════════════════════════════════════════════

print("="*65)
print("  5-Agent LightGAE -- Extended MAS Security Experiment")
print("  Goal: Demonstrate GCN structural advantage on chain attacks")
print("="*65)

# 4-1 Dataset
print("\n[1/4] 데이터 생성 (5-agent, 5 attack types + Type-V Chain)...")
N_SESS = 200
X_all, y_all, t_all = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
mask_n = (y_all == 0)
X_norm = X_all[mask_n]
atk_keys = [k for k in ATTACK_CFG if k != "Normal"]
print(f"  total={len(X_all)}  normal={mask_n.sum()}  anomaly={(~mask_n).sum()}")
print(f"  attack types: {len(atk_keys)}")

scaler   = StandardScaler().fit(X_norm.reshape(len(X_norm), -1))
X_all_s  = scaler.transform(X_all.reshape(len(X_all), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
X_norm_s = scaler.transform(X_norm.reshape(len(X_norm), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)

# Model input uses only the Core-2 columns (token_count, ctx_delta); the
# scaler above is still fit on all 5 raw columns so scaling is unaffected
# by which columns are later selected for the model.
X_all_core  = X_all_s[:, :, CORE_COLS]
X_norm_core = X_norm_s[:, :, CORE_COLS]

n_tr    = int(0.8 * len(X_norm_core))
X_train = X_norm_core[:n_tr]
X_val   = X_norm_core[n_tr:]

X_test  = np.concatenate([X_val, X_all_core[~mask_n]])
y_test  = np.concatenate([y_all[mask_n][n_tr:], y_all[~mask_n]])
t_test  = np.concatenate([t_all[mask_n][n_tr:], t_all[~mask_n]])
print(f"  train(normal)={len(X_train)}  test={len(X_test)}  (Core-2: {CORE_NAMES})")

# 4-2 Train LightGAE
print("\n[2/4] LightGAE 학습 (5-agent GCN, Core-2 input)...")
gae = LightGAE(in_dim=N_CORE, hid=16, emb=8)
train_gae(gae, X_train, epochs=150)

X_test_t = torch.FloatTensor(X_test)
sc_gae, node_sc, embeds = gae.score(X_test_t, ADJ)
tr_sc, _, _ = gae.score(torch.FloatTensor(X_train), ADJ)
theta_gae   = float(np.percentile(tr_sc, 95))
pd_gae      = (sc_gae > theta_gae).astype(int)

# 4-3 Train MLP-AE (ablation)
print("\n[3/4] MLP-AE 학습 (ablation -- no graph structure)...")
mlp = MLPAE(in_dim=N_AGENTS * N_CORE, n_feats=N_CORE, hid=16, emb=8)
train_mlp(mlp, X_train, epochs=150)
sc_mlp    = mlp.score(X_test_t)
tr_mlp    = mlp.score(torch.FloatTensor(X_train))
theta_mlp = float(np.percentile(tr_mlp, 95))
pd_mlp    = (sc_mlp > theta_mlp).astype(int)

# 4-4 Results
print("\n[4/4] 결과 비교 (GCN vs MLP)...")
r_gae = metrics(y_test, sc_gae, pd_gae)
r_mlp = metrics(y_test, sc_mlp, pd_mlp)

print(f"\n{'Method':<22} {'AUC':>7} {'F1':>7} {'TPR':>7} {'FPR':>7}")
print("─"*50)
print(f"{'LightGAE (GCN)':<22} {r_gae['AUC']:>7.4f} {r_gae['F1']:>7.4f} "
      f"{r_gae['TPR']:>7.4f} {r_gae['FPR']:>7.4f}  ◀")
print(f"{'MLP-AE (no graph)':<22} {r_mlp['AUC']:>7.4f} {r_mlp['F1']:>7.4f} "
      f"{r_mlp['TPR']:>7.4f} {r_mlp['FPR']:>7.4f}")
delta_overall = r_gae['AUC'] - r_mlp['AUC']
print("─"*50)
print(f"  ΔAUC (overall): {delta_overall:+.4f}")

# Per-attack AUC
print(f"\n  공격 유형별 ΔAUC (GCN - MLP):")
print(f"{'Attack':<22} {'LightGAE':>10} {'MLP-AE':>10} {'ΔAUC':>8}")
print("─"*52)
abl = {}
for ak in atk_keys:
    mask  = (t_test == ak) | (t_test == "Normal")
    y_s   = y_test[mask]
    auc_g = roc_auc_score(y_s, sc_gae[mask])
    auc_m = roc_auc_score(y_s, sc_mlp[mask])
    abl[ak] = {"GCN": auc_g, "MLP": auc_m, "delta": auc_g - auc_m}
    star = " ★" if ak in ("Type-V Chain", "Type-III Slow") else ""
    print(f"{ak:<22} {auc_g:>10.4f} {auc_m:>10.4f} {auc_g-auc_m:>+8.4f}{star}")

# Node-level localization
print(f"\n  에이전트별 이상 점수 (attack sessions):")
print(f"{'Attack':<22} " + "  ".join(f"{a:>14}" for a in AGENT_NAMES))
for ak in atk_keys:
    mask = (t_test == ak)
    if mask.sum() == 0: continue
    ns  = node_sc[mask].mean(axis=0)
    row = "  ".join(f"{ns[i]:>14.4f}" for i in range(N_AGENTS))
    print(f"{ak:<22} {row}")

# ══════════════════════════════════════════════════════════════
# §5.  FIGURES
# ══════════════════════════════════════════════════════════════

print("\n[Figure] 생성 중...")

# ── Fig 1: 5-Agent Graph Topology ─────────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Figure 1. Extended 5-Agent MAS: G5 = (A5, E8, M)\n"
              "vs. 3-Agent baseline G3 = (A3, E3, M)",
              fontsize=13, fontweight="bold")

G5 = nx.DiGraph()
G5.add_nodes_from(range(N_AGENTS))
for s, d in EDGES:
    G5.add_edge(s, d)
pos5 = {0:(2.0,2.0), 1:(0.5,1.0), 2:(1.5,0.0), 3:(2.5,0.0), 4:(3.5,1.0)}
ncolors5 = [BLUE, GREEN, RED, ORANGE, PURPLE]
nx.draw_networkx_nodes(G5, pos5, ax=ax1a, node_color=ncolors5, node_size=1800, alpha=0.9)
nx.draw_networkx_labels(G5, pos5, ax=ax1a,
    labels={i: AGENT_NAMES[i] for i in range(N_AGENTS)},
    font_size=8, font_weight="bold")
nx.draw_networkx_edges(G5, pos5, ax=ax1a, edge_color="#555",
    arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.12", width=1.8)
ax1a.set_title(f"(a) G5: {N_AGENTS} agents, {len(EDGES)} edges, 3-hop chain",
               fontweight="bold")
ax1a.axis("off")

G3 = nx.DiGraph()
G3.add_nodes_from([0,1,2])
for s,d in [(0,1),(1,2),(0,2)]:
    G3.add_edge(s,d)
pos3 = {0:(0.0,0.5), 1:(1.0,1.0), 2:(1.0,0.0)}
nx.draw_networkx_nodes(G3, pos3, ax=ax1b,
    node_color=[BLUE, RED, PURPLE], node_size=1800, alpha=0.9)
nx.draw_networkx_labels(G3, pos3, ax=ax1b,
    labels={0:"Orchestrator", 1:"Researcher", 2:"Writer"},
    font_size=9, font_weight="bold")
nx.draw_networkx_edges(G3, pos3, ax=ax1b, edge_color="#555",
    arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.12", width=2)
ax1b.set_title("(b) G3: 3 agents, 3 edges, 1-hop chain (baseline)",
               fontweight="bold")
ax1b.axis("off")

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_topology_g3_vs_g5.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 1 saved.")

# ── Fig 2: GCN vs MLP -- ΔAUC per Attack Type (KEY RESULT) ─────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Figure 2. GCN Structural Advantage: LightGAE vs MLP-AE per Attack Type\n"
              "(5-Agent MAS, N=200 sessions/type)",
              fontsize=12, fontweight="bold")

atk_short = [k.replace("Type-","T").replace(" ","\n") for k in abl]
gae_aucs  = [abl[k]["GCN"]   for k in abl]
mlp_aucs  = [abl[k]["MLP"]   for k in abl]
deltas    = [abl[k]["delta"] for k in abl]
x2 = np.arange(len(atk_short)); w = 0.35

bars_g = ax2a.bar(x2-w/2, gae_aucs, w, label="LightGAE (GCN)", color=RED,  alpha=0.85)
bars_m = ax2a.bar(x2+w/2, mlp_aucs, w, label="MLP-AE (no graph)", color=GRAY, alpha=0.85)
ax2a.set_xticks(x2); ax2a.set_xticklabels(atk_short, fontsize=8)
ymin = max(0.85, min(gae_aucs+mlp_aucs) - 0.03)
ax2a.set_ylim(ymin, 1.02); ax2a.grid(axis='y', alpha=0.3)
ax2a.legend(fontsize=9); ax2a.set_ylabel("AUC")
ax2a.set_title("(a) Per-Attack AUC Comparison", fontweight="bold")
for bar, v in zip(list(bars_g)+list(bars_m), gae_aucs+mlp_aucs):
    ax2a.text(bar.get_x()+bar.get_width()/2, v+0.001,
              f"{v:.4f}", ha='center', fontsize=7, fontweight='bold')

def _bar_color(k, d):
    if d > 0.003:
        return TEAL if k == "Type-V Chain" else (PURPLE if k == "Type-III Slow" else GREEN)
    if d < -0.003:
        return RED
    return GRAY

bar_colors = [_bar_color(k, abl[k]["delta"]) for k in abl]
bars_d = ax2b.bar(x2, deltas, color=bar_colors, alpha=0.85, edgecolor='white', width=0.5)
ax2b.axhline(0, color='black', linewidth=1.0)
ax2b.set_xticks(x2); ax2b.set_xticklabels(atk_short, fontsize=8)
ax2b.set_title("(b) ΔAUC = LightGAE - MLP-AE\n(green/teal = GCN wins, red = MLP wins)",
               fontweight="bold")
ax2b.set_ylabel("ΔAUC (positive = GCN wins)")
ax2b.grid(axis='y', alpha=0.3)
for bar, v in zip(bars_d, deltas):
    offset = 0.0005 if v >= 0 else -0.0015
    ax2b.text(bar.get_x()+bar.get_width()/2, v+offset,
              f"{v:+.4f}", ha='center', fontsize=9, fontweight='bold')
for i, ak in enumerate(abl):
    if abl[ak]["delta"] > 0.003:
        ax2b.annotate("★", xy=(i, deltas[i]),
                      xytext=(i, deltas[i]+0.003),
                      ha='center', fontsize=14,
                      color=TEAL if ak == "Type-V Chain" else PURPLE)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_delta_auc_gcn_vs_mlp.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 2 saved.")

# ── Fig 3: Node-Level Anomaly Heatmap ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 5))
heat_rows = {}
for ak in ["Normal"] + atk_keys:
    mask = (t_test == ak)
    if mask.sum():
        heat_rows[ak] = node_sc[mask].mean(axis=0)

heat_data  = np.array([heat_rows[k] for k in heat_rows])
row_labels = list(heat_rows.keys())
im = ax3.imshow(heat_data, aspect="auto", cmap="RdYlBu_r")
ax3.set_xticks(range(N_AGENTS)); ax3.set_xticklabels(AGENT_NAMES, fontsize=10)
ax3.set_yticks(range(len(row_labels))); ax3.set_yticklabels(row_labels, fontsize=9)
ax3.set_title("Figure 3. Per-Agent Anomaly Score Heatmap\n"
              "(5-Agent MAS -- high score = likely compromised)",
              fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax3, label="Mean Reconstruction Error")
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        ax3.text(j, i, f"{heat_data[i,j]:.2f}", ha='center', va='center',
                 fontsize=8, color="white" if heat_data[i,j] > heat_data.max()*0.6 else "black")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_node_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 3 saved.")

# ── Fig 4: ROC per Attack (GCN vs MLP) ────────────────────────
fig4, axes4 = plt.subplots(2, 3, figsize=(16, 10))
axes4 = axes4.flatten()
fig4.suptitle("Figure 4. ROC Curves per Attack Type: LightGAE (GCN) vs MLP-AE\n"
              "(5-Agent MAS -- Type-V Chain and Type-III Slow highlight GCN advantage)",
              fontsize=12, fontweight="bold")

for idx, ak in enumerate(atk_keys):
    ax = axes4[idx]
    mask  = (t_test == ak) | (t_test == "Normal")
    y_s   = y_test[mask]
    fpr_g, tpr_g, _ = roc_curve(y_s, sc_gae[mask])
    fpr_m, tpr_m, _ = roc_curve(y_s, sc_mlp[mask])
    auc_g = roc_auc_score(y_s, sc_gae[mask])
    auc_m = roc_auc_score(y_s, sc_mlp[mask])
    col   = TEAL if ak=="Type-V Chain" else (PURPLE if ak=="Type-III Slow" else RED)
    ax.plot(fpr_g, tpr_g, color=col,  lw=2.5, ls='--', label=f"LightGAE ({auc_g:.4f})")
    ax.plot(fpr_m, tpr_m, color=GRAY, lw=1.8, ls='-',  label=f"MLP-AE   ({auc_m:.4f})")
    ax.plot([0,1],[0,1],":",color="#CCC",lw=1)
    delta_str = f"ΔAUC={auc_g-auc_m:+.4f}"
    star = " ★" if ak in ("Type-V Chain","Type-III Slow") else ""
    ax.set_title(f"{ak}\n{delta_str}{star}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)

axes4[-1].axis("off")   # 5 attacks, 6 panels → hide last
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_roc_per_attack.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 4 saved.")

# ── Fig 5: Multi-Seed Robustness ──────────────────────────────
SEEDS = [42, 0, 1, 7, 123]
KEY_ATKS = ["Type-III Slow", "Type-V Chain"]
print(f"\n[다중 시드 검증] seeds={SEEDS}...")
records_gae, records_mlp = [], []
dauc_per_atk = {ak: [] for ak in KEY_ATKS}

for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    Xa, ya, ta = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
    mn = (ya == 0)
    sc_ = StandardScaler().fit(Xa[mn].reshape(mn.sum(), -1))
    Xa_s = sc_.transform(Xa.reshape(len(Xa), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xn_s = sc_.transform(Xa[mn].reshape(mn.sum(), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xa_s = Xa_s[:, :, CORE_COLS]
    Xn_s = Xn_s[:, :, CORE_COLS]
    ntr  = int(0.8 * len(Xn_s))
    Xtr, Xv = Xn_s[:ntr], Xn_s[ntr:]
    Xte = np.concatenate([Xv, Xa_s[~mn]])
    yte = np.concatenate([ya[mn][ntr:], ya[~mn]])
    tte = np.concatenate([ta[mn][ntr:], ta[~mn]])

    g = LightGAE(in_dim=N_CORE, hid=16, emb=8, verbose=False)
    train_gae(g, Xtr, epochs=100, verbose=False)
    sc_g, _, _ = g.score(torch.FloatTensor(Xte), ADJ)
    tr_g, _, _ = g.score(torch.FloatTensor(Xtr), ADJ)
    th_g = float(np.percentile(tr_g, 95))
    records_gae.append(metrics(yte, sc_g, (sc_g > th_g).astype(int)))

    mlp_m = MLPAE(in_dim=N_AGENTS*N_CORE, n_feats=N_CORE, hid=16, emb=8, verbose=False)
    train_mlp(mlp_m, Xtr, epochs=100)
    sc_m  = mlp_m.score(torch.FloatTensor(Xte))
    tr_m  = mlp_m.score(torch.FloatTensor(Xtr))
    th_m  = float(np.percentile(tr_m, 95))
    records_mlp.append(metrics(yte, sc_m, (sc_m > th_m).astype(int)))

    r_g = records_gae[-1]; r_m = records_mlp[-1]
    seed_delta = r_g['AUC'] - r_m['AUC']
    print(f"  seed={s:3d}  GCN AUC={r_g['AUC']:.4f}  MLP AUC={r_m['AUC']:.4f}  "
          f"ΔAUC(overall)={seed_delta:+.4f}", end="")

    for ak in KEY_ATKS:
        mask_ak = (tte == ak) | (tte == "Normal")
        if mask_ak.sum() > 0:
            da = roc_auc_score(yte[mask_ak], sc_g[mask_ak]) - \
                 roc_auc_score(yte[mask_ak], sc_m[mask_ak])
            dauc_per_atk[ak].append(da)
    print()

gae_means = {mk: np.mean([r[mk] for r in records_gae]) for mk in ['AUC','F1']}
gae_stds  = {mk: np.std( [r[mk] for r in records_gae]) for mk in ['AUC','F1']}
mlp_means = {mk: np.mean([r[mk] for r in records_mlp]) for mk in ['AUC','F1']}
mlp_stds  = {mk: np.std( [r[mk] for r in records_mlp]) for mk in ['AUC','F1']}
print(f"\n  LightGAE : AUC={gae_means['AUC']:.4f}±{gae_stds['AUC']:.4f}  "
      f"F1={gae_means['F1']:.4f}±{gae_stds['F1']:.4f}")
print(f"  MLP-AE   : AUC={mlp_means['AUC']:.4f}±{mlp_stds['AUC']:.4f}  "
      f"F1={mlp_means['F1']:.4f}±{mlp_stds['F1']:.4f}")
delta_seeds = [records_gae[i]['AUC'] - records_mlp[i]['AUC'] for i in range(len(SEEDS))]
print(f"  ΔAUC(overall) mean: {np.mean(delta_seeds):+.4f}  std: {np.std(delta_seeds):.4f}")
print(f"\n  Per-attack type ΔAUC across seeds:")
for ak, dlist in dauc_per_atk.items():
    star = " <<< GCN advantage" if np.mean(dlist) > 0.005 else ""
    print(f"    {ak:<20}: {np.mean(dlist):+.4f} +- {np.std(dlist):.4f}{star}")

fig5, ax5 = plt.subplots(figsize=(8, 5))
x5 = np.arange(len(SEEDS)); w5 = 0.35
g_vals = [r['AUC'] for r in records_gae]
m_vals = [r['AUC'] for r in records_mlp]
ax5.bar(x5-w5/2, g_vals, w5, label="LightGAE (GCN)", color=RED,  alpha=0.85)
ax5.bar(x5+w5/2, m_vals, w5, label="MLP-AE (no graph)", color=GRAY, alpha=0.85)
ax5.axhline(gae_means['AUC'], color=RED,  ls='--', lw=1.5, alpha=0.7,
            label=f"GCN mean={gae_means['AUC']:.4f}")
ax5.axhline(mlp_means['AUC'], color=GRAY, ls='--', lw=1.5, alpha=0.7,
            label=f"MLP mean={mlp_means['AUC']:.4f}")
ax5.set_xticks(x5); ax5.set_xticklabels([f"seed={s}" for s in SEEDS], fontsize=9)
ymin5 = max(0.85, min(g_vals+m_vals) - 0.02)
ax5.set_ylim(ymin5, 1.02); ax5.grid(axis='y', alpha=0.3)
ax5.legend(fontsize=9)
ax5.set_title(f"Figure 5. Multi-Seed AUC Robustness: LightGAE vs MLP-AE\n"
              f"(5-Agent MAS, N={N_SESS} sessions/seed)  ΔAUC={np.mean(delta_seeds):+.4f}±{np.std(delta_seeds):.4f}",
              fontsize=11, fontweight="bold")
ax5.set_ylabel("AUC")
for i, (g, m) in enumerate(zip(g_vals, m_vals)):
    ax5.text(i-w5/2, g+0.001, f"{g:.4f}", ha='center', fontsize=7, color=RED, fontweight='bold')
    ax5.text(i+w5/2, m+0.001, f"{m:.4f}", ha='center', fontsize=7, color='#555')
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_multiseed_gcn_vs_mlp.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 5 saved.")

# ══════════════════════════════════════════════════════════════
# §6.  SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  5-Agent LightGAE -- 실험 최종 요약")
print("="*65)
print(f"\n  MAS 규모:  {N_AGENTS} agents  |  {len(EDGES)} edges  |  {len(atk_keys)} attack types")
print(f"  LightGAE params: {sum(p.numel() for p in gae.parameters()):,}")
print(f"  MLPAE    params: {sum(p.numel() for p in mlp.parameters()):,}")

print(f"\n  {'Method':<22} {'AUC':>7} {'F1':>7} {'TPR':>7} {'FPR':>7}")
print("  " + "─"*50)
print(f"  {'LightGAE (GCN)':<22} {r_gae['AUC']:>7.4f} {r_gae['F1']:>7.4f} "
      f"{r_gae['TPR']:>7.4f} {r_gae['FPR']:>7.4f}  ◀")
print(f"  {'MLP-AE (no graph)':<22} {r_mlp['AUC']:>7.4f} {r_mlp['F1']:>7.4f} "
      f"{r_mlp['TPR']:>7.4f} {r_mlp['FPR']:>7.4f}")
print(f"  ΔAUC (overall): {delta_overall:+.4f}")

print(f"\n  공격 유형별 ΔAUC (핵심 결과):")
for ak, v in abl.items():
    if v['delta'] > 0.003:
        tag = " <<< GCN structural advantage"
    elif v['delta'] < -0.003:
        tag = " (MLP wins)"
    else:
        tag = ""
    print(f"    {ak:<22} {v['delta']:+.4f}{tag}")

print(f"\n  다중 시드 (N={len(SEEDS)}):")
print(f"    LightGAE : AUC={gae_means['AUC']:.4f}±{gae_stds['AUC']:.4f}")
print(f"    MLP-AE   : AUC={mlp_means['AUC']:.4f}±{mlp_stds['AUC']:.4f}")
print(f"    ΔAUC     : {np.mean(delta_seeds):+.4f}±{np.std(delta_seeds):.4f}")

print("\n  Figure 저장 위치:")
figs = ["fig1_topology_g3_vs_g5", "fig2_delta_auc_gcn_vs_mlp",
        "fig3_node_heatmap", "fig4_roc_per_attack", "fig5_multiseed_gcn_vs_mlp"]
for f in figs:
    print(f"    {OUT}/{f}.png")

print("\n실험 완료.")
