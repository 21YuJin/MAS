"""
Cross-Environment Comparison (Synthetic Simulation vs. Real-LLM) — SUPPLEMENTARY, LEGACY

이 스크립트는 headline 실험이 아니다. 최종 논문 결과는 experiments/real_llm/lgnn_experiment.py
단일 경로로만 산출한다 (README §실험 경로 참고).

여기서는 legacy 5-agent 시뮬레이션(experiments/synthetic_legacy/lgnn/multiseed_robustness_n20.py)
결과와, 이미 산출된 real-LLM headline 결과(output/real_llm/results_summary.json)를 나란히
그려볼 뿐이다. 두 숫자는 서로 다른 환경(synthetic vs. real Ollama 호출)·다른 실행에서 나온
것이므로 "하나의 최종 벤치마크 표"로 합쳐서 인용하지 말 것 — 참고용 대조일 뿐이다.

사전 조건: output/real_llm/results_summary.json 이 이미 존재해야 한다
(먼저 experiments/real_llm/lgnn_experiment.py 를 실행할 것).
"""
import json
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REAL_SUMMARY = "./output/real_llm/results_summary.json"
SIM_SUMMARY  = "./output/synthetic_legacy/lgnn_5agent/multiseed_n20_robustness.json"
OUT          = "./output/synthetic_legacy/cross_env_comparison"
os.makedirs(OUT, exist_ok=True)

BLUE = "#4C9BE8"
RED  = "#E8604C"

if not os.path.exists(REAL_SUMMARY):
    raise FileNotFoundError(
        f"{REAL_SUMMARY} 가 없습니다. 먼저 experiments/real_llm/lgnn_experiment.py 를 "
        "실행해 headline 결과를 생성하세요."
    )
if not os.path.exists(SIM_SUMMARY):
    raise FileNotFoundError(
        f"{SIM_SUMMARY} 가 없습니다. 먼저 experiments/synthetic_legacy/lgnn/"
        "multiseed_robustness_n20.py 를 실행해 legacy 시뮬레이션 결과를 생성하세요."
    )

with open(REAL_SUMMARY) as f:
    real = json.load(f)
with open(SIM_SUMMARY) as f:
    sim = json.load(f)

real_auc_mean = real["methods"]["LightGAE"]["auc_mean"]
real_auc_std  = real["methods"]["LightGAE"]["auc_std"]
sim_auc_mean  = float(np.mean(sim["gcn_auc_per_seed"]))
sim_auc_std   = float(np.std(sim["gcn_auc_per_seed"]))

gap = sim_auc_mean - real_auc_mean

print("[Cross-Environment Comparison — supplementary, not headline]")
print(f"  Synthetic simulation (legacy, N={sim['n_seeds']} seeds) AUC: "
      f"{sim_auc_mean:.4f} +/- {sim_auc_std:.4f}")
print(f"  Real-LLM (headline, N={len(real['seeds'])} seeds)          AUC: "
      f"{real_auc_mean:.4f} +/- {real_auc_std:.4f}")
print(f"  Gap (sim - real): {gap:+.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
envs = [f"Synthetic (legacy)\nN={sim['n_seeds']} seeds", f"Real LLM (headline)\nN={len(real['seeds'])} seeds"]
means = [sim_auc_mean, real_auc_mean]
stds  = [sim_auc_std, real_auc_std]
bars = ax.bar(envs, means, yerr=stds, capsize=6, color=[BLUE, RED], alpha=0.85, width=0.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel("LightGAE AUC")
ax.set_title("Cross-Environment Comparison (Supplementary — NOT a unified benchmark)",
              fontsize=11, fontweight="bold")
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}",
            ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/cross_env_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

with open(f"{OUT}/cross_env_comparison.json", "w") as f:
    json.dump({
        "note": "supplementary comparison, not the headline result",
        "synthetic_legacy_auc_mean": sim_auc_mean,
        "synthetic_legacy_auc_std": sim_auc_std,
        "real_llm_auc_mean": real_auc_mean,
        "real_llm_auc_std": real_auc_std,
        "gap_sim_minus_real": gap,
    }, f, indent=2)

print(f"\n저장 완료: {OUT}/cross_env_comparison.png, {OUT}/cross_env_comparison.json")
