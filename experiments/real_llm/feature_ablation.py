"""
Feature-set ablation on the real-LLM (Ollama llama3.2) 4-agent data, reusing
the cached sessions from lgnn_experiment.py (output/real_llm/cache_*.json,
already reordered to [latency, token_count, ctx_delta, sentence_count,
joint_deviation_flag]).

Same purpose as experiments/lgnn/feature_ablation_5agent.py, run on the
simulation: check whether the 2 "extension" features (sentence_count,
joint_deviation_flag) add detection value on top of the 3 "core" features
(latency, token_count, ctx_delta), using real LLM execution data instead of
synthetic data.
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

N_AGENTS    = 4
AGENT_NAMES = ["Orchestrator", "Researcher", "Analyst", "Writer"]
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
N_FEATS     = len(FEAT_NAMES)
EDGES       = [(0, 1), (1, 2), (2, 3), (0, 2)]

def build_adj(n_agents=N_AGENTS, edges=EDGES):
    A = np.zeros((n_agents, n_agents), dtype=np.float32)
    for s, d in edges:
        A[s, d] = A[d, s] = 1.0
    A += np.eye(n_agents, dtype=np.float32)
    deg  = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)

ADJ = build_adj()

# ══════════════════════════════════════════════════════════════
# Load cached sessions (same cache lgnn_experiment.py uses)
# ══════════════════════════════════════════════════════════════

OUT = "./output/real_llm"
CACHE_NORMAL = os.path.join(OUT, "cache_normal.json")
CACHE_ATTACK = os.path.join(OUT, "cache_attack.json")

with open(CACHE_NORMAL) as f:
    X_normal = np.array(json.load(f), dtype=np.float32)   # (50, 4, 5)
with open(CACHE_ATTACK) as f:
    X_attack = np.array(json.load(f), dtype=np.float32)   # (50, 4, 5)

N_NORMAL = len(X_normal)
N_ATTACK = len(X_attack)
print(f"Loaded cache: normal={N_NORMAL}  attack={N_ATTACK}")

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


def train_gae(model, X_normal, A, epochs=160, lr=1e-3, bs=16):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t   = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i+bs]]
            X_hat = model(b, A)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def metrics(y, sc, pred):
    if len(np.unique(y)) < 2:
        return dict(TPR=0, FPR=0, F1=0, AUC=0.5)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        TPR=tp/(tp+fn+1e-8), FPR=fp/(fp+tn+1e-8),
        F1=f1_score(y, pred, zero_division=0),
        AUC=roc_auc_score(y, sc),
    )

# ══════════════════════════════════════════════════════════════
# Ablation driver
# ══════════════════════════════════════════════════════════════

FEATURE_SETS = {
    "Full-5 (all)":                    [0,1,2,3,4],
    "Core-3 (latency,token,ctx_delta)": [0,1,2],
    "Core-2 (token,ctx_delta)":        [1,2],
    "Full-5 minus latency":            [1,2,3,4],
    "Full-5 minus token_count":        [0,2,3,4],
    "Full-5 minus ctx_delta":          [0,1,3,4],
    "Full-5 minus sentence_count":     [0,1,2,4],
    "Full-5 minus joint_deviation":    [0,1,2,3],
}

SEEDS = [42, 0, 1, 7, 123]
n_tr  = int(N_NORMAL * 0.80)   # 40

print("="*70)
print("  Feature-Set Ablation -- Real-LLM 4-Agent MAS (Ollama llama3.2)")
print("="*70)

results = {name: {"AUC": [], "F1": []} for name in FEATURE_SETS}

for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)

    idx_n     = np.random.permutation(N_NORMAL)
    Xn_sh_raw = X_normal[idx_n]
    X_tr_raw  = Xn_sh_raw[:n_tr]
    X_val_raw = Xn_sh_raw[n_tr:]

    scaler   = StandardScaler().fit(X_tr_raw.reshape(len(X_tr_raw), -1))
    X_tr_all = scaler.transform(X_tr_raw.reshape(len(X_tr_raw), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    X_val_all= scaler.transform(X_val_raw.reshape(len(X_val_raw), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xa_all   = scaler.transform(X_attack.reshape(N_ATTACK, -1)).reshape(N_ATTACK, N_AGENTS, N_FEATS).astype(np.float32)
    X_te_all = np.concatenate([X_val_all, Xa_all])
    y_te     = np.array([0]*len(X_val_all) + [1]*N_ATTACK)

    print(f"\n  seed={s}")
    for name, cols in FEATURE_SETS.items():
        X_tr = X_tr_all[:, :, cols]
        X_te = X_te_all[:, :, cols]

        model = LightGAE(in_dim=len(cols), hid=16, emb=8)
        train_gae(model, X_tr, ADJ, epochs=160, lr=1e-3, bs=16)

        sc_test = model.score(torch.FloatTensor(X_te), ADJ)
        sc_tr   = model.score(torch.FloatTensor(X_tr), ADJ)
        theta   = float(np.percentile(sc_tr, 95))
        pred    = (sc_test > theta).astype(int)

        m = metrics(y_te, sc_test, pred)
        results[name]["AUC"].append(m["AUC"])
        results[name]["F1"].append(m["F1"])
        print(f"    {name:<38} AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  Multi-seed summary (N=5 seeds)")
print("="*70)
print(f"  {'Feature set':<38} {'AUC':>17} {'F1':>17}")
print("  " + "-"*72)
full5_auc = np.mean(results["Full-5 (all)"]["AUC"])
for name in FEATURE_SETS:
    a = np.array(results[name]["AUC"]); f = np.array(results[name]["F1"])
    tag = "" if name == "Full-5 (all)" else f"  (ΔAUC vs Full-5: {a.mean()-full5_auc:+.4f})"
    print(f"  {name:<38} {a.mean():.4f}±{a.std():.4f}  {f.mean():.4f}±{f.std():.4f}{tag}")

t_c3, p_c3 = stats.ttest_rel(results["Core-3 (latency,token,ctx_delta)"]["F1"], results["Full-5 (all)"]["F1"])
print(f"\n  [paired t-test, F1, N=5 seeds] Core-3 vs Full-5: t={t_c3:+.3f}  p={p_c3:.4f}")

# ══════════════════════════════════════════════════════════════
# Feature correlation matrix (raw, normal sessions)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  Feature correlation matrix (Pearson, raw values, normal sessions)")
print("="*70)
flat = X_normal.reshape(-1, N_FEATS)
corr = np.corrcoef(flat, rowvar=False)
header = " ".join(f"{n[:10]:>10}" for n in FEAT_NAMES)
print(f"  {'':<16}{header}")
for i, name in enumerate(FEAT_NAMES):
    row = " ".join(f"{corr[i,j]:>10.3f}" for j in range(N_FEATS))
    print(f"  {name:<16}{row}")
max_off = max(abs(corr[i,j]) for i in range(N_FEATS) for j in range(N_FEATS) if i != j)
print(f"\n  최대 |off-diagonal correlation| = {max_off:.3f}")

print("\n실험 완료.")
