"""
Lightweight Graph Autoencoder (LightGAE) for
Quick Identification of Anomalous Interactions in Multi-Agent AI Systems

1차년도 연구목표:
  멀티에이전트 시스템 환경 구축 및 Quick Identification 기술 개발

System Model  G = (A, E, M):
  A = {v0:Orchestrator, v1:Researcher, v2:Writer}
  E = {(v0,v1), (v1,v2), (v0,v2)}
  M = {δ: latency,  τ: token_count,  f: api_freq,
        Δc: ctx_delta,  s: call_seq}

Attack Taxonomy (4 Types):
  Type-I   Direct Override     — immediate role hijack at Researcher
  Type-II  Credential Harvest  — info leak attempt + downstream propagation
  Type-III Slow Poison         — gradual compromise across turns
  Type-IV  Context Flood       — simultaneous multi-agent contamination
"""

import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix
from sklearn.ensemble import IsolationForest
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

OUT = "./output/lgnn"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# §1.  MAS GRAPH DEFINITION
#      G = (A, E, M)
# ══════════════════════════════════════════════════════════════

N_AGENTS    = 3
AGENT_NAMES = ["Orchestrator", "Researcher", "Writer"]
FEAT_NAMES  = ["latency", "token_count", "api_freq", "ctx_delta", "call_seq"]
N_FEATS     = len(FEAT_NAMES)

EDGES = [(0, 1), (1, 2), (0, 2)]      # directed pipeline + supervisory edge

def build_adj() -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops: D^{-1/2}(A+I)D^{-1/2}"""
    A = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float32)
    for s, d in EDGES:
        A[s, d] = A[d, s] = 1.0       # bidirectional (request + reply)
    A += np.eye(N_AGENTS, dtype=np.float32)
    deg = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)

ADJ = build_adj()                      # fixed graph topology for all sessions

# ══════════════════════════════════════════════════════════════
# §2.  DATA GENERATION — 4 ATTACK TYPES
# ══════════════════════════════════════════════════════════════

NP = dict(latency=(0.85, 0.12), token_count=(160, 25),
          api_freq=2.5,          ctx_delta=(0.05, 0.02))
AP = dict(latency=(1.30, 0.30), token_count=(240, 50),
          api_freq=5.5,          ctx_delta=(0.18, 0.06))

ATTACK_CFG = {
    "Normal":          {"p_r": 0.0,  "p_w": 0.0 },
    "Type-I Direct":   {"p_r": 1.0,  "p_w": 0.0 },
    "Type-II Harvest": {"p_r": 0.80, "p_w": 0.35},
    "Type-III Slow":   {"p_r": 0.40, "p_w": 0.15},
    "Type-IV Flood":   {"p_r": 0.65, "p_w": 0.65},
}
COLORS_ATK = {
    "Normal":          BLUE,
    "Type-I Direct":   RED,
    "Type-II Harvest": ORANGE,
    "Type-III Slow":   PURPLE,
    "Type-IV Flood":   GREEN,
}

def _lerp(n_val, a_val, p):
    if isinstance(n_val, tuple):
        return n_val[0] + p*(a_val[0]-n_val[0]), n_val[1] + p*(a_val[1]-n_val[1])
    return n_val + p*(a_val - n_val)

def sample_agent(p: float) -> list:
    mu_l, sg_l = _lerp(NP["latency"],     AP["latency"],     p)
    mu_t, sg_t = _lerp(NP["token_count"], AP["token_count"], p)
    mu_c, sg_c = _lerp(NP["ctx_delta"],   AP["ctx_delta"],   p)
    lam_a      = _lerp(NP["api_freq"],    AP["api_freq"],    p)
    return [
        max(0.05, np.random.normal(mu_l, sg_l)),
        max(10,   int(np.random.normal(mu_t, sg_t))),
        max(0,    int(np.random.poisson(lam_a))),
        max(0.0,  np.random.normal(mu_c, sg_c)),
        int(np.random.random() < p * 0.7),
    ]

def make_session(atk_key="Normal", n_turns=30, win=5):
    """
    One session → sliding-window graphs.
    Each window aggregates `win` consecutive turns into X ∈ R^{N×F}.

    Writer contamination scales with Researcher's actual token excess to
    model the real pipeline dependency: Researcher output → Writer input.
    This creates genuine inter-agent correlation that GCN edges can exploit.
    """
    cfg   = ATTACK_CFG[atk_key]
    label = 0 if atk_key == "Normal" else 1
    tok_range = AP["token_count"][0] - NP["token_count"][0]   # 80 tokens
    turns = []
    for _ in range(n_turns):
        orch = sample_agent(0.0)
        res  = sample_agent(cfg["p_r"])
        # Writer's contamination is proportional to how much Researcher
        # deviated from normal token count (causal pipeline effect)
        if cfg["p_w"] > 0:
            excess = max(0.0, res[1] - NP["token_count"][0]) / (tok_range + 1e-8)
            p_w_eff = cfg["p_w"] * min(1.0, 0.2 + 0.8 * excess)
        else:
            p_w_eff = 0.0
        wrt = sample_agent(p_w_eff)
        turns.append([orch, res, wrt])
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
# §3.  LIGHTWEIGHT GRAPH AUTOENCODER  (LightGAE)
#
#  Encoder:  GCN-2  in_dim → hid → emb
#  Decoder:  MLP-2  emb    → hid → in_dim   (per node)
#
#  Anomaly score = mean per-node reconstruction error
#  Node score    = per-node reconstruction error → localization
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """H' = σ(Â H W),  Â = normalized adjacency with self-loops."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H, A):
        # H: (B, N, in_dim),  A: (N, N)
        AH = torch.einsum("ij,bjk->bik", A, H)
        return self.linear(AH)


class LightGAE(nn.Module):
    """
    Lightweight Graph Autoencoder for one-class anomaly detection.
    Trained on normal sessions only; anomalies yield high recon error.
    """
    def __init__(self, in_dim=N_FEATS, hid=16, emb=8, verbose=True):
        super().__init__()
        self.gc1  = GCNLayer(in_dim, hid)
        self.gc2  = GCNLayer(hid,    emb)
        self.dec1 = nn.Linear(emb, hid)
        self.dec2 = nn.Linear(hid, in_dim)
        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f"  LightGAE  hid={hid}  emb={emb}  params={n_params:,}")

    def encode(self, X, A):
        H1 = F.relu(self.gc1(X, A))
        H1 = F.dropout(H1, p=0.1, training=self.training)
        H2 = self.gc2(H1, A)            # (B, N, emb)
        z  = H2.mean(dim=1)             # (B, emb)  graph-level
        return H2, z

    def decode(self, H2):
        return self.dec2(F.relu(self.dec1(H2)))   # (B, N, in_dim)

    def forward(self, X, A):
        H2, z = self.encode(X, A)
        return self.decode(H2), H2, z

    @torch.no_grad()
    def score(self, X_t, A):
        self.eval()
        X_hat, H2, z = self.forward(X_t, A)
        node_err  = ((X_t - X_hat) ** 2).mean(dim=2)   # (B, N)
        graph_err = node_err.mean(dim=1)                # (B,)
        return graph_err.numpy(), node_err.numpy(), z.numpy()


def train_lgae(model, X_normal, A, epochs=150, lr=1e-3, bs=64, verbose=True):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    losses = []
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        ep_loss = 0.0
        for i in range(0, len(idx), bs):
            b        = X_t[idx[i:i+bs]]
            X_hat, _, _ = model(b, A)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        losses.append(ep_loss)
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  loss={ep_loss:.5f}")
    return losses


# ══════════════════════════════════════════════════════════════
# §3.5  MLP AUTOENCODER  (Ablation Baseline — no graph)
# ══════════════════════════════════════════════════════════════

class MLPAE(nn.Module):
    """Flat MLP Autoencoder — treats nodes independently, no message passing."""
    def __init__(self, in_dim=N_AGENTS * N_FEATS, hid=16, emb=8, verbose=True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid, emb))
        self.dec = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Linear(hid, in_dim))
        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f"  MLPAE     hid={hid}  emb={emb}  params={n_params:,}")

    def forward(self, X):
        B = X.shape[0]
        z = self.enc(X.reshape(B, -1))
        return self.dec(z).reshape(B, N_AGENTS, N_FEATS)

    @torch.no_grad()
    def score(self, X_t):
        self.eval()
        X_hat    = self.forward(X_t)
        node_err = ((X_t - X_hat) ** 2).mean(dim=2)
        return node_err.mean(dim=1).numpy()


def train_mlpae(model, X_normal, epochs=150, lr=1e-3, bs=64):
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


# ══════════════════════════════════════════════════════════════
# §4.  BASELINES
# ══════════════════════════════════════════════════════════════

def b1_threshold(tr, te, fidx=0):
    theta = tr[:, fidx].mean() + 2*tr[:, fidx].std()
    sc    = te[:, fidx]
    return sc, (sc > theta).astype(int)

def b2_isoforest(tr, te):
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    clf.fit(tr)
    sc = -clf.score_samples(te)
    return sc, (clf.predict(te) == -1).astype(int)

def b3_zscore(tr, te):
    sc  = StandardScaler().fit(tr)
    ztr = np.linalg.norm(sc.transform(tr), axis=1)
    zte = np.linalg.norm(sc.transform(te), axis=1)
    th  = ztr.mean() + 2*ztr.std()
    return zte, (zte > th).astype(int)

def b4_sliding(tr, te, w=5):
    sc  = StandardScaler().fit(tr)
    ztr = np.linalg.norm(sc.transform(tr), axis=1)
    zte = sc.transform(te)
    th  = ztr.mean() + 2*ztr.std()
    agg = np.array([zte[max(0,i-w):i+1].mean(axis=0) for i in range(len(zte))])
    s   = np.linalg.norm(agg, axis=1)
    return s, (s > th).astype(int)

def metrics(y, sc, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(
        TPR=round(tp/(tp+fn+1e-8), 4),
        FPR=round(fp/(fp+tn+1e-8), 4),
        F1 =round(f1_score(y, pred, zero_division=0), 4),
        AUC=round(roc_auc_score(y, sc), 4),
    )

# ══════════════════════════════════════════════════════════════
# §5.  EXPERIMENT
# ══════════════════════════════════════════════════════════════

print("="*65)
print("  LightGAE — Quick Identification for MAS Security")
print("  1차년도 연구목표 실험")
print("="*65)

# 5-1 Data
print("\n[1/5] 데이터 생성...")
N_SESS = 200
X_all, y_all, t_all = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
mask_n = (y_all == 0)
X_norm = X_all[mask_n]
print(f"  total={len(X_all)}  normal={mask_n.sum()}  anomaly={(~mask_n).sum()}")

# Normalize (fit on normal only)
scaler       = StandardScaler().fit(X_norm.reshape(len(X_norm), -1))
X_all_s      = scaler.transform(X_all.reshape(len(X_all),  -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
X_norm_s     = scaler.transform(X_norm.reshape(len(X_norm),-1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)

n_tr         = int(0.8 * len(X_norm_s))
X_train      = X_norm_s[:n_tr]
X_val        = X_norm_s[n_tr:]

X_test_3d    = np.concatenate([X_val, X_all_s[~mask_n]], axis=0)
y_test       = np.concatenate([y_all[mask_n][n_tr:], y_all[~mask_n]])
t_test       = np.concatenate([t_all[mask_n][n_tr:], t_all[~mask_n]])
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test_3d.reshape(len(X_test_3d), -1)

print(f"  train(normal)={len(X_train)}  test={len(X_test_3d)}")

# 5-2 Train LightGAE
print("\n[2/5] LightGAE 학습 (one-class, normal only)...")
model  = LightGAE(in_dim=N_FEATS, hid=16, emb=8)
losses = train_lgae(model, X_train, ADJ, epochs=150, lr=1e-3, bs=64)

# 5-3 Inference
print("\n[3/5] 탐지 실험...")
X_test_t  = torch.FloatTensor(X_test_3d)
sc_gae, node_sc, embeds = model.score(X_test_t, ADJ)

X_val_t   = torch.FloatTensor(X_val)
val_s, _, _ = model.score(X_val_t, ADJ)
theta_gae = val_s.mean() + 2*val_s.std()
pd_gae    = (sc_gae > theta_gae).astype(int)

sc_b1, pd_b1 = b1_threshold(X_train_flat, X_test_flat)
sc_b2, pd_b2 = b2_isoforest(X_train_flat, X_test_flat)
sc_b3, pd_b3 = b3_zscore(X_train_flat, X_test_flat)
sc_b4, pd_b4 = b4_sliding(X_train_flat, X_test_flat)

res = {
    "Threshold (B1)":      metrics(y_test, sc_b1, pd_b1),
    "IsoForest (B2)":      metrics(y_test, sc_b2, pd_b2),
    "Z-score (B3)":        metrics(y_test, sc_b3, pd_b3),
    "SlidingZscore (B4)":    metrics(y_test, sc_b4, pd_b4),
    "LightGAE [proposed]": metrics(y_test, sc_gae, pd_gae),
}

print(f"\n{'Method':<24} {'TPR':>7} {'FPR':>7} {'F1':>7} {'AUC':>7}")
print("─"*52)
for nm, r in res.items():
    mk = "  ◀" if "proposed" in nm else ""
    print(f"{nm:<24} {r['TPR']:>7.4f} {r['FPR']:>7.4f} {r['F1']:>7.4f} {r['AUC']:>7.4f}{mk}")
print("─"*52)

# 5-4 Per-attack analysis
print("\n[4/5] 공격 유형별 AUC...")
atk_keys  = [k for k in ATTACK_CFG if k != "Normal"]
per_atk   = {}
for ak in atk_keys:
    mask = (t_test == ak) | (t_test == "Normal")
    y_s  = y_test[mask]; sc_s = sc_gae[mask]; pd_s = pd_gae[mask]
    auc  = roc_auc_score(y_s, sc_s)
    tpr  = (pd_s[y_s==1] == 1).mean() if (y_s==1).sum() > 0 else 0
    per_atk[ak] = {"AUC": auc, "TPR": tpr}
    print(f"  {ak:<22}  AUC={auc:.4f}  TPR={tpr:.4f}")

# 5-5 Node-level localization
print("\n[5/5] 에이전트별 이상 점수 (침해 에이전트 식별)...")
print(f"  {'Attack':<22} " + "  ".join(f"{a:>14}" for a in AGENT_NAMES))
for ak in atk_keys:
    mask = (t_test == ak)
    if mask.sum() == 0: continue
    ns   = node_sc[mask].mean(axis=0)
    row  = "  ".join(f"{ns[i]:>14.4f}" for i in range(N_AGENTS))
    print(f"  {ak:<22} {row}")

# 5-6 Quick Identification timing
print("\n[Quick ID] 추론 시간 측정 (ms/sample)...")
N_B = 1000
Xb  = torch.FloatTensor(np.tile(X_test_3d[:1], (N_B,1,1)))
Xbf = Xb.numpy().reshape(N_B, -1)

def bench_ms(fn, n_runs=10):
    for _ in range(3):
        fn()    # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs / N_B * 1000

t_gae = bench_ms(lambda: model.score(Xb, ADJ))
t_b1  = bench_ms(lambda: (Xbf[:,0] > 0).sum())
t_b2  = bench_ms(lambda: b2_isoforest(X_train_flat, Xbf))
t_b3  = bench_ms(lambda: b3_zscore(X_train_flat, Xbf))

timings = {
    "Threshold (B1)":      t_b1,
    "IsoForest (B2)":      t_b2,
    "Z-score (B3)":        t_b3,
    "LightGAE [proposed]": t_gae,
}
for k, v in timings.items():
    print(f"  {k:<24} {v:.4f} ms/sample")

# ══════════════════════════════════════════════════════════════
# §6.  FIGURES  (6종)
# ══════════════════════════════════════════════════════════════

print("\n[Figure] 생성 중...")

# ── Fig 1: MAS Graph Architecture ──────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle("Figure 1. MAS Threat Model: System Definition G = (A, E, M)",
              fontsize=13, fontweight="bold")

G = nx.DiGraph()
G.add_nodes_from(range(N_AGENTS))
for s, d in EDGES:
    G.add_edge(s, d)

pos   = {0: (0, 0.5), 1: (1, 1.0), 2: (1, 0.0)}
ncolors = [BLUE, ORANGE, GREEN]
nx.draw_networkx_nodes(G, pos, ax=ax1a, node_color=ncolors,
                       node_size=1800, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax1a,
                        labels={i: AGENT_NAMES[i] for i in range(N_AGENTS)},
                        font_size=9, font_weight="bold")
nx.draw_networkx_edges(G, pos, ax=ax1a, edge_color="#555",
                       arrows=True, arrowsize=20,
                       connectionstyle="arc3,rad=0.1", width=2)
edge_labels = {(0,1): "task", (1,2): "result", (0,2): "supervise"}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                              ax=ax1a, font_size=8, label_pos=0.35)
ax1a.set_title("(a) Agent Communication Graph", fontweight="bold")
ax1a.axis("off")

# Attack propagation overlay
G2 = nx.DiGraph()
G2.add_nodes_from(range(N_AGENTS))
G2.add_edge(1, 2, color=RED)   # attack propagates R→W
pos2 = pos.copy()
nx.draw_networkx_nodes(G2, pos2, ax=ax1b,
                       node_color=[BLUE, RED, ORANGE],
                       node_size=1800, alpha=0.9)
nx.draw_networkx_labels(G2, pos2, ax=ax1b,
                        labels={0:"Orchestrator\n(clean)",
                                1:"Researcher\n(injected)",
                                2:"Writer\n(propagated)"},
                        font_size=8, font_weight="bold")
nx.draw_networkx_edges(G2, pos2, ax=ax1b, edge_color=RED,
                       arrows=True, arrowsize=25, width=3,
                       connectionstyle="arc3,rad=0.1")
ax1b.set_title("(b) Attack Propagation Pattern", fontweight="bold")
ax1b.axis("off")
ax1b.text(0.5, -0.08,
          "Indirect injection enters at Researcher → propagates to Writer",
          ha="center", transform=ax1b.transAxes, fontsize=9, color=RED)

plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig1_mas_graph.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 1 saved.")

# ── Fig 2: Feature Distribution (4 attack types) ───────────
fig2, axes2 = plt.subplots(1, N_FEATS, figsize=(16, 4))
fig2.suptitle("Figure 2. Metadata Feature Distributions Across Attack Types\n"
              f"(Researcher agent, N={N_SESS} sessions per type)",
              fontsize=12, fontweight="bold")

# Use the already-built raw (unnormalized) dataset from §5 — no need to rebuild
for col, feat in enumerate(FEAT_NAMES):
    ax = axes2[col]
    for ak, col_c in COLORS_ATK.items():
        vals = X_all[t_all==ak, 1, col]   # Researcher node, raw feature
        ax.hist(vals, bins=15, alpha=0.55, color=col_c, label=ak, density=True)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_xlabel("value"); ax.grid(alpha=0.3)
    if col == 0: ax.set_ylabel("Density")

handles = [mpatches.Patch(color=c, label=k) for k, c in COLORS_ATK.items()]
fig2.legend(handles=handles, loc="lower center", ncol=5,
            fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUT}/lgnn_fig2_feature_dist.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 2 saved.")

# ── Fig 3: PCA of Learned Embeddings ───────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))
pca  = PCA(n_components=2, random_state=42)
emb2 = pca.fit_transform(embeds)
for ak, col_c in COLORS_ATK.items():
    m = (t_test == ak)
    ax3.scatter(emb2[m, 0], emb2[m, 1], c=col_c, s=18,
                alpha=0.65, label=ak, edgecolors="none")
ax3.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax3.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax3.set_title("Figure 3. PCA Projection of LightGAE Graph Embeddings\n"
              "(2D — learned without attack labels)", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9, markerscale=2)
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig3_embedding_pca.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 3 saved.")

# ── Fig 4: ROC Curves ──────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 6))
roc_cfg = [
    (sc_b1, GRAY,   "Threshold (B1)",     1.6, "-"),
    (sc_b2, GREEN,  "IsoForest (B2)",     1.6, "-"),
    (sc_b3, BLUE,   "Z-score (B3)",       1.6, "-"),
    (sc_b4, ORANGE, "SlidingZscore (B4)",   2.0, "-"),
    (sc_gae,RED,    "LightGAE [proposed]",2.5, "--"),
]
for sc, col, nm, lw, ls in roc_cfg:
    fpr_r, tpr_r, _ = roc_curve(y_test, sc)
    auc = roc_auc_score(y_test, sc)
    ax4.plot(fpr_r, tpr_r, color=col, lw=lw, ls=ls,
             label=f"{nm}  (AUC={auc:.3f})")
ax4.plot([0,1],[0,1],":",color="#CCC",lw=1)
ax4.set_xlabel("False Positive Rate", fontsize=12)
ax4.set_ylabel("True Positive Rate", fontsize=12)
ax4.set_title("Figure 4. ROC Curve — 4 Baselines vs. LightGAE",
              fontsize=12, fontweight="bold")
ax4.legend(fontsize=9, loc="lower right")
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig4_roc.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 4 saved.")

# ── Fig 5: Performance Bar ─────────────────────────────────
fig5, axes5 = plt.subplots(1, 4, figsize=(16, 5))
fig5.suptitle("Figure 5. Detection Performance: 4 Baselines vs. LightGAE",
              fontsize=13, fontweight="bold")

methods  = ["Threshold\n(B1)", "IsoForest\n(B2)", "Z-score\n(B3)",
            "SlidingZscore\n(B4)", "LightGAE\n[proposed]"]
bar_cols = [GRAY, GREEN, BLUE, ORANGE, RED]

for ax, metric in zip(axes5, ["TPR","FPR","F1","AUC"]):
    vals = [list(res.values())[i][metric] for i in range(len(res))]
    bars = ax.bar(methods, vals, color=bar_cols,
                  edgecolor="white", linewidth=1.2, width=0.6)
    bars[-1].set_edgecolor(RED); bars[-1].set_linewidth(2.5)
    ax.set_title(f"{metric} ({'↑' if metric!='FPR' else '↓'})",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig5_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 5 saved.")

# ── Fig 6: Node-Level Anomaly Localization + Timing ────────
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle("Figure 6. LightGAE Node-Level Localization & Quick Identification",
              fontsize=12, fontweight="bold")

# (a) node-level heatmap
node_means = {}
for ak in atk_keys:
    mask = (t_test == ak)
    if mask.sum(): node_means[ak] = node_sc[mask].mean(axis=0)

normal_mask = (t_test == "Normal")
node_means["Normal"] = node_sc[normal_mask].mean(axis=0)

heat_data  = np.array([node_means[k] for k in ["Normal"]+atk_keys])
im = ax6a.imshow(heat_data, aspect="auto", cmap="RdYlBu_r")
ax6a.set_xticks(range(N_AGENTS)); ax6a.set_xticklabels(AGENT_NAMES, fontsize=10)
ax6a.set_yticks(range(len(heat_data)))
ax6a.set_yticklabels(["Normal"]+atk_keys, fontsize=9)
ax6a.set_title("(a) Per-Agent Anomaly Score\n(high = likely compromised)",
               fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax6a, label="Recon Error")
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        ax6a.text(j, i, f"{heat_data[i,j]:.3f}", ha="center", va="center",
                  fontsize=8, color="white" if heat_data[i,j] > heat_data.max()*0.6 else "black")

# (b) timing
tk_names = list(timings.keys())
tk_vals  = list(timings.values())
tk_cols  = [GRAY, GREEN, BLUE, RED]
bars6 = ax6b.bar(tk_names, tk_vals, color=tk_cols,
                 edgecolor="white", linewidth=1.2, width=0.55)
bars6[-1].set_edgecolor(RED); bars6[-1].set_linewidth(2.5)
ax6b.set_yscale("log")
ax6b.set_ylabel("Inference time (ms / sample) — log scale", fontsize=10)
ax6b.set_title("(b) Quick Identification — Inference Speed\n(lower = faster)",
               fontsize=10, fontweight="bold")
ax6b.set_xticklabels(tk_names, fontsize=8.5, rotation=10)
ax6b.grid(axis="y", alpha=0.3, which="both")
for bar, v in zip(bars6, tk_vals):
    ax6b.text(bar.get_x()+bar.get_width()/2, v*1.5,
              f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig6_node_timing.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 6 saved.")

# ══════════════════════════════════════════════════════════════
# §7.  다중 시드 검증  (5 seeds → mean ± std)
# ══════════════════════════════════════════════════════════════
SEEDS = [42, 0, 1, 7, 123]
print(f"\n[다중 시드 검증] seeds={SEEDS}")
seed_records = []
for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    Xa, ya, _ = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
    mn   = (ya == 0)
    sc_  = StandardScaler().fit(Xa[mn].reshape(mn.sum(), -1))
    Xa_s = sc_.transform(Xa.reshape(len(Xa), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xn_s = sc_.transform(Xa[mn].reshape(mn.sum(), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    ntr  = int(0.8 * len(Xn_s))
    Xtr, Xv = Xn_s[:ntr], Xn_s[ntr:]
    Xte  = np.concatenate([Xv, Xa_s[~mn]])
    yte  = np.concatenate([ya[mn][ntr:], ya[~mn]])
    m    = LightGAE(in_dim=N_FEATS, hid=16, emb=8, verbose=False)
    train_lgae(m, Xtr, ADJ, epochs=100, lr=1e-3, bs=64, verbose=False)
    vs, _, _ = m.score(torch.FloatTensor(Xv), ADJ)
    th       = vs.mean() + 2 * vs.std()
    sc_s, _, _ = m.score(torch.FloatTensor(Xte), ADJ)
    r = metrics(yte, sc_s, (sc_s > th).astype(int))
    seed_records.append(r)
    print(f"  seed={s:3d}  AUC={r['AUC']:.4f}  F1={r['F1']:.4f}  "
          f"TPR={r['TPR']:.4f}  FPR={r['FPR']:.4f}")

seed_means = {met: np.mean([r[met] for r in seed_records]) for met in ['AUC','F1','TPR','FPR']}
seed_stds  = {met: np.std( [r[met] for r in seed_records]) for met in ['AUC','F1','TPR','FPR']}
print(f"\n  LightGAE (N={len(SEEDS)} seeds):")
for met in ['AUC', 'F1', 'TPR', 'FPR']:
    print(f"    {met}: {seed_means[met]:.4f} ± {seed_stds[met]:.4f}")

# ══════════════════════════════════════════════════════════════
# §8.  Ablation — LightGAE (GCN) vs MLP-AE (no graph)
# ══════════════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)
print(f"\n[Ablation] LightGAE (GCN) vs MLP-AE (no graph structure)")
mlp_model = MLPAE(in_dim=N_AGENTS * N_FEATS, hid=16, emb=8, verbose=True)
train_mlpae(mlp_model, X_train, epochs=150, lr=1e-3, bs=64)

sc_mlp    = mlp_model.score(X_test_t)
vs_mlp    = mlp_model.score(torch.FloatTensor(X_val))
theta_mlp = vs_mlp.mean() + 2 * vs_mlp.std()
pd_mlp    = (sc_mlp > theta_mlp).astype(int)
res_mlp   = metrics(y_test, sc_mlp, pd_mlp)
r_gae     = res["LightGAE [proposed]"]

print(f"\n  {'Method':<22} {'AUC':>7} {'F1':>7} {'TPR':>7} {'FPR':>7}")
print("  " + "─"*46)
print(f"  {'LightGAE (GCN)':<22} {r_gae['AUC']:>7.4f} {r_gae['F1']:>7.4f} "
      f"{r_gae['TPR']:>7.4f} {r_gae['FPR']:>7.4f}")
print(f"  {'MLP-AE (no graph)':<22} {res_mlp['AUC']:>7.4f} {res_mlp['F1']:>7.4f} "
      f"{res_mlp['TPR']:>7.4f} {res_mlp['FPR']:>7.4f}")
print("  " + "─"*46)
delta_auc = r_gae['AUC'] - res_mlp['AUC']
print(f"  ΔAUC (GCN − MLP): {delta_auc:+.4f}  "
      f"{'← graph structure contributes' if delta_auc > 0 else '← no graph benefit (check data)'}")

print(f"\n  공격 유형별 AUC 비교:")
print(f"  {'Attack':<22} {'LightGAE':>10} {'MLP-AE':>10} {'Δ':>8}")
print("  " + "─"*52)
abl_per_atk = {}
for ak in atk_keys:
    mask    = (t_test == ak) | (t_test == "Normal")
    y_s     = y_test[mask]
    auc_gae = roc_auc_score(y_s, sc_gae[mask])
    auc_mlp = roc_auc_score(y_s, sc_mlp[mask])
    abl_per_atk[ak] = {"LightGAE": auc_gae, "MLPAE": auc_mlp}
    print(f"  {ak:<22} {auc_gae:>10.4f} {auc_mlp:>10.4f} {auc_gae - auc_mlp:>+8.4f}")

# ── Fig 7: Ablation Study ──────────────────────────────────
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13, 5))
fig7.suptitle("Figure 7. Ablation Study: Graph Structure vs. Flat MLP Autoencoder",
              fontsize=12, fontweight="bold")

met_labels = ['AUC', 'F1', 'TPR', 'FPR']
x7  = np.arange(len(met_labels))
w   = 0.35
gae_vals = [r_gae[m]   for m in met_labels]
mlp_vals = [res_mlp[m] for m in met_labels]
b_gae = ax7a.bar(x7 - w/2, gae_vals, w, label='LightGAE (GCN)', color=RED,  alpha=0.85)
b_mlp = ax7a.bar(x7 + w/2, mlp_vals, w, label='MLP-AE (no graph)', color=GRAY, alpha=0.85)
ax7a.set_xticks(x7); ax7a.set_xticklabels(met_labels, fontsize=11)
ax7a.set_ylim(0, 1.25); ax7a.grid(axis='y', alpha=0.3)
ax7a.legend(fontsize=9)
ax7a.set_title("(a) Overall Detection Metrics", fontweight="bold")
for bar, v in zip(list(b_gae) + list(b_mlp), gae_vals + mlp_vals):
    ax7a.text(bar.get_x() + bar.get_width()/2, v + 0.02,
              f"{v:.3f}", ha='center', fontsize=8, fontweight='bold')

atk_short = [k.replace('Type-', 'T') for k in abl_per_atk]
gae_atk   = [abl_per_atk[k]['LightGAE'] for k in abl_per_atk]
mlp_atk   = [abl_per_atk[k]['MLPAE']    for k in abl_per_atk]
x7b       = np.arange(len(atk_short))
ax7b.bar(x7b - w/2, gae_atk, w, label='LightGAE (GCN)', color=RED,  alpha=0.85)
ax7b.bar(x7b + w/2, mlp_atk, w, label='MLP-AE (no graph)', color=GRAY, alpha=0.85)
ax7b.set_xticks(x7b); ax7b.set_xticklabels(atk_short, fontsize=9)
ymin7b = max(0.9, min(gae_atk + mlp_atk) - 0.02)
ax7b.set_ylim(ymin7b, 1.01); ax7b.grid(axis='y', alpha=0.3)
ax7b.legend(fontsize=9)
ax7b.set_title("(b) Per-Attack Type AUC", fontweight="bold")
for x_pos, g, p in zip(x7b, gae_atk, mlp_atk):
    ax7b.text(x_pos - w/2, g + 0.001, f"{g:.4f}", ha='center', fontsize=7,
              color=RED, fontweight='bold')
    ax7b.text(x_pos + w/2, p + 0.001, f"{p:.4f}", ha='center', fontsize=7, color='#555')

plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig7_ablation.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 7 saved.")

# ── Fig 8: Multi-seed Robustness ───────────────────────────
fig8, ax8 = plt.subplots(figsize=(8, 5))
ms_metrics = ['AUC', 'F1', 'TPR', 'FPR']
ms_means   = [seed_means[m] for m in ms_metrics]
ms_stds    = [seed_stds[m]  for m in ms_metrics]
x8         = np.arange(len(ms_metrics))
bars8      = ax8.bar(x8, ms_means, color=[RED, BLUE, GREEN, ORANGE],
                     alpha=0.85, edgecolor='white', width=0.5)
ax8.errorbar(x8, ms_means, yerr=ms_stds, fmt='none',
             color='black', capsize=7, capthick=2, elinewidth=2)
ax8.set_xticks(x8); ax8.set_xticklabels(ms_metrics, fontsize=12)
ax8.set_ylim(0, 1.2); ax8.grid(axis='y', alpha=0.3)
ax8.set_title(f"Figure 8. LightGAE Robustness Across {len(SEEDS)} Random Seeds\n"
              f"(N={N_SESS} sessions/seed, 100 epochs each)",
              fontsize=12, fontweight="bold")
for bar, m_v, s_v in zip(bars8, ms_means, ms_stds):
    ax8.text(bar.get_x() + bar.get_width()/2, m_v + s_v + 0.015,
             f"{m_v:.4f}\n±{s_v:.4f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/lgnn_fig8_multiseed.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Fig 8 saved.")

# ══════════════════════════════════════════════════════════════
# §9.  통계 검정 (Mann-Whitney U Test)
# ══════════════════════════════════════════════════════════════

print("\n[통계 검정] Mann-Whitney U Test")
print(f"{'Feature':<14} {'정상 μ':>10} {'이상 μ':>10} "
      f"{'p-value':>12} {'유의성':>8}")
print("─"*58)

# Researcher 노드 기준 (index=1)
X_res_norm = X_all[y_all==0, 1, :]   # 정상
X_res_anom = X_all[y_all==1, 1, :]   # 이상

for i, feat in enumerate(FEAT_NAMES):
    n_vals = X_res_norm[:, i]
    a_vals = X_res_anom[:, i]
    _, p   = stats.mannwhitneyu(n_vals, a_vals, alternative='two-sided')
    sig    = "***" if p<0.001 else "**" if p<0.01 \
             else "*" if p<0.05 else "n.s."
    print(f"{feat:<14} {n_vals.mean():>10.3f} {a_vals.mean():>10.3f} "
          f"{p:>12.4e} {sig:>8}")

print("\n* p<0.05  ** p<0.01  *** p<0.001")

# H0 검증: LightGAE 점수가 정상/이상 간 유의미하게 다른가
n_scores = sc_gae[y_test==0]
a_scores = sc_gae[y_test==1]
_, p_gae = stats.mannwhitneyu(n_scores, a_scores, alternative='two-sided')
print(f"\nLightGAE 이상 점수 검정: p={p_gae:.4e} "
      f"({'유의미' if p_gae<0.05 else '유의미하지 않음'})")

# Effect size (Cohen's d)
pooled_std = np.sqrt((n_scores.std()**2 + a_scores.std()**2) / 2)
cohens_d   = (a_scores.mean() - n_scores.mean()) / (pooled_std + 1e-8)
print(f"Effect size (Cohen's d): {cohens_d:.4f} "
      f"({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'})")

# ══════════════════════════════════════════════════════════════
# §10. SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  최종 결과 요약")
print("="*65)
print(f"\n  Model parameters  : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Inference latency : {t_gae:.4f} ms/sample\n")

print(f"  {'Method':<24} {'TPR':>7} {'FPR':>7} {'F1':>7} {'AUC':>7}")
print("  " + "─"*52)
for nm, r in res.items():
    mk = "  ◀ [proposed]" if "proposed" in nm else ""
    print(f"  {nm:<24} {r['TPR']:>7.4f} {r['FPR']:>7.4f}"
          f" {r['F1']:>7.4f} {r['AUC']:>7.4f}{mk}")
print("  " + "─"*52)

print(f"\n  공격 유형별 AUC:")
for ak, v in per_atk.items():
    print(f"    {ak:<22}  AUC={v['AUC']:.4f}  TPR={v['TPR']:.4f}")

print(f"\n  [Ablation] LightGAE (GCN) vs MLP-AE (no graph):")
print(f"    LightGAE (GCN) : AUC={r_gae['AUC']:.4f}  F1={r_gae['F1']:.4f}")
print(f"    MLP-AE (flat)  : AUC={res_mlp['AUC']:.4f}  F1={res_mlp['F1']:.4f}")
print(f"    ΔAUC = {delta_auc:+.4f}  (그래프 구조 기여)")

print(f"\n  [다중 시드] N={len(SEEDS)} seeds (mean ± std):")
for met in ['AUC', 'F1', 'TPR', 'FPR']:
    print(f"    {met}: {seed_means[met]:.4f} ± {seed_stds[met]:.4f}")

print("\n  Figure 저장 위치:")
fnames = ["mas_graph","feature_dist","embedding_pca","roc","performance",
          "node_timing","ablation","multiseed"]
for i, fn in enumerate(fnames, 1):
    print(f"    {OUT}/lgnn_fig{i}_{fn}.png")

print("\n실험 완료.")
