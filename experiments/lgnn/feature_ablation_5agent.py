"""
Feature-set ablation on the 5-agent MAS (same data-generation process as
mas_lgnn_5agent.py, reused here to keep results directly comparable).

Goal: empirically justify the 5-feature set
  [latency, token_count, ctx_delta, sentence_count, joint_deviation_flag]
by testing whether:
  (A) the 2 "extension" features (sentence_count, joint_deviation_flag) add
      detection value on top of the 3 "core" features (latency, token_count,
      ctx_delta) -- Core-3 vs Full-5, multi-seed.
  (B) each individual feature is non-redundant -- leave-one-out from Full-5,
      multi-seed. A feature that can be dropped with no AUC/F1 loss would be
      a candidate for removal; one whose removal hurts performance is
      empirically justified.
  (C) the 5 features are not simply restating the same signal -- Pearson
      correlation matrix on raw (unscaled) normal-session values.

This does NOT retrain a "3-feature-only production model" -- it is a
diagnostic ablation to support the feature-selection narrative in the paper.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# Reuse the exact data-generation process from mas_lgnn_5agent.py
# ══════════════════════════════════════════════════════════════

N_AGENTS    = 5
AGENT_NAMES = ["Orchestrator", "Planner", "Researcher", "Analyst", "Writer"]
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
N_FEATS     = len(FEAT_NAMES)

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,2),(0,3),(0,4),(1,3)]

def build_adj() -> torch.Tensor:
    A = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float32)
    for s, d in EDGES:
        A[s,d] = A[d,s] = 1.0
    A += np.eye(N_AGENTS, dtype=np.float32)
    deg  = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)

ADJ = build_adj()

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

def _lerp(n_val, a_val, p):
    if isinstance(n_val, tuple):
        return n_val[0]+p*(a_val[0]-n_val[0]), n_val[1]+p*(a_val[1]-n_val[1])
    return n_val + p*(a_val - n_val)

def sample_agent(p: float, context_scale: float = 1.0) -> list:
    mu_l, sg_l = _lerp(NP["latency"],     AP["latency"],     p)
    mu_t, sg_t = _lerp(NP["token_count"], AP["token_count"], p)
    mu_c, sg_c = _lerp(NP["ctx_delta"],   AP["ctx_delta"],   p)
    lam_a      = _lerp(NP["sentence_count"], AP["sentence_count"], p)
    lat_scale  = 0.5 + 0.5 * context_scale
    ctx_scale  = 0.3 + 0.7 * context_scale
    lat_val = max(0.05, np.random.normal(mu_l * lat_scale, sg_l))
    tok_val = max(10,   int(np.random.normal(mu_t, sg_t)))
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
    cfg     = ATTACK_CFG[atk_key]
    label   = 0 if atk_key == "Normal" else 1
    tok_rng = AP["token_count"][0] - NP["token_count"][0]

    def eff_p(p_base, upstream_tokens):
        if p_base <= 0:
            return 0.0
        excess = max(0.0, upstream_tokens - NP["token_count"][0]) / (tok_rng + 1e-8)
        return p_base * min(1.0, 0.2 + 0.8 * excess)

    def upstream_signal(feats):
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

def build_dataset(n_sess=200, n_turns=30, win=5):
    Xs, ys, ts = [], [], []
    for ak in ATTACK_CFG:
        for _ in range(n_sess):
            for X, y, t in make_session(ak, n_turns, win):
                Xs.append(X); ys.append(y); ts.append(t)
    return np.array(Xs), np.array(ys), np.array(ts)

# ══════════════════════════════════════════════════════════════
# Models (feature-dim agnostic)
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, H, A):
        return self.linear(torch.einsum("ij,bjk->bik", A, H))


class LightGAE(nn.Module):
    def __init__(self, in_dim, hid=16, emb=8):
        super().__init__()
        self.gc1  = GCNLayer(in_dim, hid)
        self.gc2  = GCNLayer(hid,    emb)
        self.dec1 = nn.Linear(emb, hid)
        self.dec2 = nn.Linear(hid, in_dim)

    def encode(self, X, A):
        H1 = F.relu(self.gc1(X, A))
        H1 = F.dropout(H1, p=0.1, training=self.training)
        return self.gc2(H1, A)

    def forward(self, X, A):
        H2 = self.encode(X, A)
        return self.dec2(F.relu(self.dec1(H2)))

    @torch.no_grad()
    def score(self, X_t, A):
        self.eval()
        X_hat = self.forward(X_t, A)
        node_err  = ((X_t - X_hat) ** 2).mean(dim=2)
        return node_err.mean(dim=1).numpy()


def train_gae(model, X_normal, epochs=100, lr=1e-3, bs=64):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i+bs]]
            X_hat = model(b, ADJ)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def metrics(y, sc, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(
        TPR=tp/(tp+fn+1e-8), FPR=fp/(fp+tn+1e-8),
        F1=f1_score(y, pred, zero_division=0),
        AUC=roc_auc_score(y, sc),
    )

# ══════════════════════════════════════════════════════════════
# Ablation driver: given a full 5-feature dataset, evaluate LightGAE
# restricted to a subset of feature columns.
# ══════════════════════════════════════════════════════════════

FEATURE_SETS = {
    "Full-5 (all)":                    [0,1,2,3,4],
    "Core-3 (latency,token,ctx_delta)": [0,1,2],
    "Full-5 minus latency":            [1,2,3,4],
    "Full-5 minus token_count":        [0,2,3,4],
    "Full-5 minus ctx_delta":          [0,1,3,4],
    "Full-5 minus sentence_count":     [0,1,2,4],
    "Full-5 minus joint_deviation":    [0,1,2,3],
}

SEEDS = [42, 0, 1, 7, 123]
N_SESS = 200

print("="*70)
print("  Feature-Set Ablation -- 5-Agent MAS (justifying the 5-feature set)")
print("="*70)

results = {name: {"AUC": [], "F1": []} for name in FEATURE_SETS}

for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    X_all, y_all, t_all = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
    mask_n = (y_all == 0)

    scaler   = StandardScaler().fit(X_all[mask_n].reshape(mask_n.sum(), -1))
    X_all_s  = scaler.transform(X_all.reshape(len(X_all), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    X_norm_s = X_all_s[mask_n]

    n_tr    = int(0.8 * len(X_norm_s))
    X_train_full = X_norm_s[:n_tr]
    X_val_full   = X_norm_s[n_tr:]
    X_test_full  = np.concatenate([X_val_full, X_all_s[~mask_n]])
    y_test       = np.concatenate([y_all[mask_n][n_tr:], y_all[~mask_n]])

    print(f"\n  seed={s}")
    for name, cols in FEATURE_SETS.items():
        X_train = X_train_full[:, :, cols]
        X_test  = X_test_full[:, :, cols]

        model = LightGAE(in_dim=len(cols), hid=16, emb=8)
        train_gae(model, X_train, epochs=100)

        sc_test = model.score(torch.FloatTensor(X_test), ADJ)
        sc_tr   = model.score(torch.FloatTensor(X_train), ADJ)
        theta   = float(np.percentile(sc_tr, 95))
        pred    = (sc_test > theta).astype(int)

        m = metrics(y_test, sc_test, pred)
        results[name]["AUC"].append(m["AUC"])
        results[name]["F1"].append(m["F1"])
        print(f"    {name:<38} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

# ══════════════════════════════════════════════════════════════
# Summary: multi-seed mean +/- std per feature set
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  (A)+(B) Multi-seed summary (N=5 seeds)")
print("="*70)
print(f"  {'Feature set':<38} {'AUC':>17} {'F1':>17}")
print("  " + "-"*72)
full5_auc = np.mean(results["Full-5 (all)"]["AUC"])
full5_f1  = np.mean(results["Full-5 (all)"]["F1"])
for name in FEATURE_SETS:
    a = np.array(results[name]["AUC"]); f = np.array(results[name]["F1"])
    d_auc = a.mean() - full5_auc
    tag = ""
    if name != "Full-5 (all)":
        tag = f"  (ΔAUC vs Full-5: {d_auc:+.4f})"
    print(f"  {name:<38} {a.mean():.4f}±{a.std():.4f}  {f.mean():.4f}±{f.std():.4f}{tag}")

# ══════════════════════════════════════════════════════════════
# (C) Feature correlation matrix (raw, unscaled, normal sessions only)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  (C) Feature correlation matrix (Pearson, raw values, normal sessions)")
print("="*70)
np.random.seed(42); torch.manual_seed(42)
X_all, y_all, t_all = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
X_norm_raw = X_all[y_all == 0]                      # (N, 5 agents, 5 feats)
flat = X_norm_raw.reshape(-1, N_FEATS)               # pool all agents/sessions
corr = np.corrcoef(flat, rowvar=False)

header = " ".join(f"{n[:10]:>10}" for n in FEAT_NAMES)
print(f"  {'':<16}{header}")
for i, name in enumerate(FEAT_NAMES):
    row = " ".join(f"{corr[i,j]:>10.3f}" for j in range(N_FEATS))
    print(f"  {name:<16}{row}")

max_off_diag = max(abs(corr[i,j]) for i in range(N_FEATS) for j in range(N_FEATS) if i != j)
print(f"\n  최대 |off-diagonal correlation| = {max_off_diag:.3f}  "
      f"({'낮음 -> feature 간 중복성 낮음' if max_off_diag < 0.5 else '주의: 일부 feature 쌍이 상관관계 높음'})")

print("\n실험 완료.")
