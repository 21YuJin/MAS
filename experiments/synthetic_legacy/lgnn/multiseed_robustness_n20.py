"""
[LEGACY / SYNTHETIC — not the headline experiment]
Headline results now come exclusively from experiments/real_llm/lgnn_experiment.py.
This script uses synthetic (non-LLM) simulated data and is kept for reference only.

N=20-seed robustness check for the GCN (LightGAE) vs MLP-AE structural
advantage claim on the 5-agent MAS simulation, Core-2 features
(token_count, ctx_delta).

Motivation: the N=5-seed check in mas_lgnn_5agent.py gave a paired t-test
that was significant (p=0.0326) under one package-version combination but
not significant (p=0.089) under another (same code, same seeds, different
torch/numpy versions) -- see README history. This script re-runs the same
comparison with more seeds and reports a fuller set of robustness
diagnostics (bootstrap CI, sign-consistency ratio, permutation test) so the
conclusion doesn't rest on a single point-estimate p-value.

Data-generation and model code duplicated from mas_lgnn_5agent.py (same
convention as feature_ablation_5agent.py) so this script has no side effects
on the headline script/figures.
"""

import json
import os
import sys
import warnings

import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = "./output/lgnn_5agent"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# Data generation (identical to mas_lgnn_5agent.py)
# ══════════════════════════════════════════════════════════════

N_AGENTS    = 5
AGENT_NAMES = ["Orchestrator", "Planner", "Researcher", "Analyst", "Writer"]
FEAT_NAMES  = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]
N_FEATS     = len(FEAT_NAMES)
CORE_COLS   = [1, 2]   # token_count, ctx_delta -- final headline feature set
N_CORE      = len(CORE_COLS)

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (0, 3), (0, 4), (1, 3)]


def build_adj() -> torch.Tensor:
    A = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float32)
    for s, d in EDGES:
        A[s, d] = A[d, s] = 1.0
    A += np.eye(N_AGENTS, dtype=np.float32)
    deg = A.sum(axis=1)
    dinv = np.diag(1.0 / np.sqrt(deg + 1e-8))
    return torch.FloatTensor(dinv @ A @ dinv)


ADJ = build_adj()

NP = dict(latency=(0.85, 0.12), token_count=(160, 25),
          sentence_count=2.5,    ctx_delta=(0.05, 0.02))
AP = dict(latency=(1.30, 0.30), token_count=(240, 50),
          sentence_count=5.5,    ctx_delta=(0.18, 0.06))

ATTACK_CFG = {
    "Normal":          {"p_pln": 0.00, "p_res": 0.00, "p_ana": 0.00, "p_wrt": 0.00},
    "Type-I Direct":   {"p_pln": 0.00, "p_res": 1.00, "p_ana": 0.00, "p_wrt": 0.00},
    "Type-II Harvest": {"p_pln": 0.00, "p_res": 0.80, "p_ana": 0.35, "p_wrt": 0.00},
    "Type-III Slow":   {"p_pln": 0.00, "p_res": 0.30, "p_ana": 0.10, "p_wrt": 0.05},
    "Type-IV Flood":   {"p_pln": 0.00, "p_res": 0.65, "p_ana": 0.50, "p_wrt": 0.65},
    "Type-V Chain":    {"p_pln": 0.40, "p_res": 0.00, "p_ana": 0.00, "p_wrt": 0.00},
}


def _lerp(n_val, a_val, p):
    if isinstance(n_val, tuple):
        return n_val[0] + p * (a_val[0] - n_val[0]), n_val[1] + p * (a_val[1] - n_val[1])
    return n_val + p * (a_val - n_val)


def sample_agent(p: float, context_scale: float = 1.0) -> list:
    mu_l, sg_l = _lerp(NP["latency"], AP["latency"], p)
    mu_t, sg_t = _lerp(NP["token_count"], AP["token_count"], p)
    mu_c, sg_c = _lerp(NP["ctx_delta"], AP["ctx_delta"], p)
    lam_a = _lerp(NP["sentence_count"], AP["sentence_count"], p)
    lat_scale = 0.5 + 0.5 * context_scale
    ctx_scale = 0.3 + 0.7 * context_scale
    lat_val = max(0.05, np.random.normal(mu_l * lat_scale, sg_l))
    tok_val = max(10, int(np.random.normal(mu_t, sg_t)))
    lat_z = (lat_val - NP["latency"][0]) / NP["latency"][1]
    tok_z = (tok_val - NP["token_count"][0]) / NP["token_count"][1]
    return [
        lat_val,
        tok_val,
        max(0.0, np.random.normal(mu_c * ctx_scale, sg_c)),
        max(0, int(np.random.poisson(lam_a))),
        int(lat_z > 1.5 and tok_z > 1.0),
    ]


def make_session(atk_key="Normal", n_turns=30, win=5):
    cfg = ATTACK_CFG[atk_key]
    label = 0 if atk_key == "Normal" else 1
    tok_rng = AP["token_count"][0] - NP["token_count"][0]

    def eff_p(p_base, upstream_tokens):
        if p_base <= 0:
            return 0.0
        excess = max(0.0, upstream_tokens - NP["token_count"][0]) / (tok_rng + 1e-8)
        return p_base * min(1.0, 0.2 + 0.8 * excess)

    def upstream_signal(feats):
        return max(feats[1] / NP["token_count"][0], feats[0] / NP["latency"][0])

    turns = []
    for _ in range(n_turns):
        orch = sample_agent(0.0)
        pln = sample_agent(cfg["p_pln"])
        pln_sig = upstream_signal(pln)
        p_res = eff_p(cfg["p_res"], pln[1]) if cfg["p_pln"] > 0 else cfg["p_res"]
        res = sample_agent(p_res, context_scale=pln_sig)
        res_sig = upstream_signal(res)
        ana = sample_agent(eff_p(cfg["p_ana"], res[1]), context_scale=res_sig)
        ana_sig = upstream_signal(ana)
        wrt = sample_agent(eff_p(cfg["p_wrt"], ana[1]), context_scale=ana_sig)
        turns.append([orch, pln, res, ana, wrt])

    out = []
    for i in range(win, n_turns + 1):
        X = np.mean(turns[i - win:i], axis=0).astype(np.float32)
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
# Models (identical to mas_lgnn_5agent.py)
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
        self.gc1 = GCNLayer(in_dim, hid)
        self.gc2 = GCNLayer(hid, emb)
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
        return ((X_t - X_hat) ** 2).mean(dim=2).mean(dim=1).numpy()


class MLPAE(nn.Module):
    def __init__(self, in_dim, n_feats, hid=16, emb=8):
        super().__init__()
        self.n_feats = n_feats
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, emb))
        self.dec = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, in_dim))

    def forward(self, X):
        B = X.shape[0]
        return self.dec(self.enc(X.reshape(B, -1))).reshape(B, N_AGENTS, self.n_feats)

    @torch.no_grad()
    def score(self, X_t):
        self.eval()
        return ((X_t - self.forward(X_t)) ** 2).mean(dim=2).mean(dim=1).numpy()


def train_gae(model, X_normal, epochs=100, lr=1e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i + bs]]
            X_hat = model(b, ADJ)
            loss = F.mse_loss(X_hat, b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def train_mlp(model, X_normal, epochs=100, lr=1e-3, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X_t = torch.FloatTensor(X_normal)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_t))
        for i in range(0, len(idx), bs):
            b = X_t[idx[i:i + bs]]
            loss = F.mse_loss(model(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def metrics(y, sc, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        TPR=tp / (tp + fn + 1e-8), FPR=fp / (fp + tn + 1e-8),
        F1=f1_score(y, pred, zero_division=0),
        AUC=roc_auc_score(y, sc),
    )


# ══════════════════════════════════════════════════════════════
# N=20-seed run
# ══════════════════════════════════════════════════════════════

SEEDS = [42, 0, 1, 7, 123, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
assert len(SEEDS) == 20
N_SESS = 200

print("=" * 70)
print(f"  N={len(SEEDS)}-seed robustness check: LightGAE (GCN) vs MLP-AE, Core-2")
print("=" * 70)

gcn_auc, mlp_auc = [], []
for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    Xa, ya, ta = build_dataset(n_sess=N_SESS, n_turns=30, win=5)
    mn = (ya == 0)
    sc_ = StandardScaler().fit(Xa[mn].reshape(mn.sum(), -1))
    Xa_s = sc_.transform(Xa.reshape(len(Xa), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xn_s = sc_.transform(Xa[mn].reshape(mn.sum(), -1)).reshape(-1, N_AGENTS, N_FEATS).astype(np.float32)
    Xa_s = Xa_s[:, :, CORE_COLS]
    Xn_s = Xn_s[:, :, CORE_COLS]
    ntr = int(0.8 * len(Xn_s))
    Xtr, Xv = Xn_s[:ntr], Xn_s[ntr:]
    Xte = np.concatenate([Xv, Xa_s[~mn]])
    yte = np.concatenate([ya[mn][ntr:], ya[~mn]])

    g = LightGAE(in_dim=N_CORE, hid=16, emb=8)
    train_gae(g, Xtr, epochs=100)
    sc_g = g.score(torch.FloatTensor(Xte), ADJ)
    tr_g = g.score(torch.FloatTensor(Xtr), ADJ)
    th_g = float(np.percentile(tr_g, 95))
    r_g = metrics(yte, sc_g, (sc_g > th_g).astype(int))

    m = MLPAE(in_dim=N_AGENTS * N_CORE, n_feats=N_CORE, hid=16, emb=8)
    train_mlp(m, Xtr, epochs=100)
    sc_m = m.score(torch.FloatTensor(Xte))
    tr_m = m.score(torch.FloatTensor(Xtr))
    th_m = float(np.percentile(tr_m, 95))
    r_m = metrics(yte, sc_m, (sc_m > th_m).astype(int))

    gcn_auc.append(r_g["AUC"]); mlp_auc.append(r_m["AUC"])
    print(f"  seed={s:4d}  GCN AUC={r_g['AUC']:.4f}  MLP AUC={r_m['AUC']:.4f}  "
          f"ΔAUC={r_g['AUC'] - r_m['AUC']:+.4f}")

gcn_auc = np.array(gcn_auc); mlp_auc = np.array(mlp_auc)
delta = gcn_auc - mlp_auc

# ── Diagnostics ─────────────────────────────────────────────────
mean_delta = float(delta.mean())
sd_delta   = float(delta.std(ddof=1))
pos_ratio  = float((delta > 0).mean())

t_stat, p_ttest = stats.ttest_rel(gcn_auc, mlp_auc, alternative="two-sided")

rng = np.random.default_rng(0)
N_BOOT = 10000
boot_means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(N_BOOT)])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

N_PERM = 10000
signs = rng.choice([-1, 1], size=(N_PERM, len(delta)))
perm_means = (signs * delta).mean(axis=1)
p_perm = float((np.abs(perm_means) >= abs(mean_delta)).mean())

env_versions = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
}

print("\n" + "=" * 70)
print(f"  Summary (N={len(SEEDS)} seeds)")
print("=" * 70)
print(f"  Environment           : {env_versions}")
print(f"  GCN AUC per seed      : {[round(float(x), 4) for x in gcn_auc]}")
print(f"  MLP AUC per seed      : {[round(float(x), 4) for x in mlp_auc]}")
print(f"  ΔAUC per seed         : {[round(float(x), 4) for x in delta]}")
print(f"  Mean ΔAUC             : {mean_delta:+.4f}")
print(f"  Sample SD (ddof=1)    : {sd_delta:.4f}")
print(f"  95% bootstrap CI      : [{ci_lo:+.4f}, {ci_hi:+.4f}]  (n_boot={N_BOOT})")
print(f"  Positive-seed ratio   : {pos_ratio:.2f}  ({int((delta>0).sum())}/{len(delta)} seeds GCN > MLP)")
print(f"  Paired t-test         : t={t_stat:+.4f}, p={p_ttest:.6f}")
print(f"  Sign-flip permutation : p={p_perm:.6f}  (n_perm={N_PERM})")

result = {
    "env_versions": env_versions,
    "seeds": SEEDS,
    "gcn_auc_per_seed": [float(x) for x in gcn_auc],
    "mlp_auc_per_seed": [float(x) for x in mlp_auc],
    "delta_auc_per_seed": [float(x) for x in delta],
    "mean_delta_auc": mean_delta,
    "sd_delta_auc_ddof1": sd_delta,
    "bootstrap_ci95": [float(ci_lo), float(ci_hi)],
    "n_bootstrap": N_BOOT,
    "positive_seed_ratio": pos_ratio,
    "n_seeds": len(SEEDS),
    "paired_ttest": {"t_statistic": float(t_stat), "p_value_two_sided": float(p_ttest)},
    "permutation_test": {"p_value_two_sided": p_perm, "n_permutations": N_PERM, "method": "sign_flip"},
}
out_path = f"{OUT}/multiseed_n20_robustness.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {out_path}")
