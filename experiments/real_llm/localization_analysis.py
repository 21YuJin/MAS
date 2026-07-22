"""
Propagation ground-truth schema + node-level localization metrics (step 15,
first half of step 17's "Node-level" bullet). No new Ollama calls: reads
already-collected data --
  - configs/attacks/chain.json               (attack template -> injection_agent /
                                                expected_propagation_path)
  - output/real_llm/session_metadata_attack.json  (attack_type per session)
  - output/real_llm/results_summary.json          (per_seed[].methods.LightGAE.
                                                     test_node_scores, added in step 17)
and joins them -- it does not retrain any model.

Two things this script CANNOT do, and does not pretend to:
  1. observed_propagation_path -- real_llm_v1's cache_attack.json stores only the
     5-dim extracted feature vector per agent, not the raw response text, so there
     is no way to retroactively detect (from these 50 sessions) which downstream
     agents actually complied with the injected instructions vs. which only
     received elevated token counts as a side effect. Left null with an explicit
     "unavailable" reason per session -- populate this at the NEXT collection by
     saving raw text (or a per-node compliance flag derived from it) alongside
     the feature vector.
  2. A non-degenerate affected-node benchmark -- every one of the 7 ATTACK_TYPES
     slugs in the current dataset is generated from configs/attacks/chain.json, and
     all 7 templates share expected_propagation_path=[all 4 nodes]. So
     "affected-node Hit@1" and "score_ratio" are trivial/undefined on this dataset
     (the affected set covers the whole graph, and no session has an "unaffected"
     node to contrast against). entry-node top-1/MRR/mean-rank are NOT trivial --
     see the finding printed at the bottom of main(): the entry node is picked
     top-1 only ~2% of the time, because LightGAE's node score currently tracks
     downstream cascade *impact*, not the injection *entry point*. A real
     affected-node benchmark needs attack templates with a partial propagation
     path / varied target_agent (configs/attacks/direct.json, slow.json,
     length_preserving.json already exist for this, but aren't wired into
     collection yet -- see README §공격 시나리오).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.topology import load_agent_names  # noqa: E402  [Step 1-3] single-sourced from topology config

OUT = "./output/real_llm"
CHAIN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "attacks", "chain.json")
META_ATTACK_PATH  = os.path.join(OUT, "session_metadata_attack.json")
RESULTS_PATH      = os.path.join(OUT, "results_summary.json")

AGENT_NAMES = load_agent_names()   # topology_4agent_v1 node order (was a hardcoded literal)


def load_chain_templates():
    with open(CHAIN_CONFIG_PATH, encoding="utf-8") as f:
        templates = json.load(f)
    by_attack_type = {}
    for t in templates:
        # template_id is "chain_v1_<attack_type>" -- attack_type (session provenance
        # metadata) is the ATTACK_TYPES slug from lgnn_experiment.py, template_id
        # (attack config) is that same slug with the "chain_v1_" template-version
        # prefix. Assert the join actually lines up rather than silently no-op-ing
        # if the naming convention ever drifts.
        assert t["template_id"].startswith("chain_v1_"), t["template_id"]
        attack_type = t["template_id"][len("chain_v1_"):]
        by_attack_type[attack_type] = t
    return by_attack_type


def build_propagation_ground_truth(meta_attack, templates_by_type):
    """Schema (step 15): one record per attack session.
    {
      "session_id", "attack_type", "template_id",
      "injection_entry_node", "directly_compromised_nodes",
      "propagation_affected_nodes", "unaffected_nodes",
      "expected_propagation_path", "observed_propagation_path" (null here -- see
      module docstring), "observed_propagation_note"
    }
    directly_compromised_nodes / propagation_affected_nodes / unaffected_nodes
    are all DERIVED from expected_propagation_path + injection_entry_node -- not
    independently re-specified -- so they can't drift out of sync with it.
    """
    records = []
    for m in meta_attack:
        atype = m["attack_type"]
        tmpl = templates_by_type[atype]
        entry = tmpl["injection_agent"]
        expected_path = tmpl["expected_propagation_path"]
        assert entry == expected_path[0], \
            f"{atype}: injection_agent {entry!r} must be the first hop of expected_propagation_path {expected_path!r}"
        directly_compromised = [entry]
        propagation_affected = [n for n in expected_path if n != entry]
        unaffected = [n for n in AGENT_NAMES if n not in expected_path]
        records.append({
            "session_id": m["session_id"],
            "attack_type": atype,
            "template_id": tmpl["template_id"],
            "injection_entry_node": entry,
            "directly_compromised_nodes": directly_compromised,
            "propagation_affected_nodes": propagation_affected,
            "unaffected_nodes": unaffected,
            "expected_propagation_path": expected_path,
            "observed_propagation_path": None,
            "observed_propagation_note": "unavailable -- real_llm_v1 cache stores only "
                "the extracted feature vector per agent, not raw response text, so "
                "response-based propagation cannot be retrofit onto sessions already "
                "collected. Populate at the next collection (save raw text or a "
                "per-node compliance flag alongside the feature vector).",
        })
    return records


def node_localization_metrics(attack_node_scores, prop_gt, agent_names=AGENT_NAMES):
    """attack_node_scores: (n_attack, n_agents) LightGAE reconstruction error,
    row-aligned with prop_gt (both in meta_attack/session order).
    Per session: rank agents by descending score, then:
      - entry_rank         : 1-indexed rank of the true injection_entry_node
      - entry_top1         : entry_rank == 1
      - entry_reciprocal_rank : 1 / entry_rank   (-> MRR when averaged)
      - affected_hit_at_1   : is the top-1-scored node in the affected set
                               (directly_compromised + propagation_affected)?
      - score_ratio         : mean(affected-node scores) / mean(unaffected-node
                               scores + eps) -- None when a session has no
                               unaffected nodes (can't form the ratio's
                               denominator; NOT reported as 0 or 1).
    """
    per_session = []
    for scores, gt in zip(attack_node_scores, prop_gt):
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)   # descending
        ranked_agents = [agent_names[i] for i in order]
        entry = gt["injection_entry_node"]
        entry_rank = ranked_agents.index(entry) + 1
        affected = set(gt["directly_compromised_nodes"]) | set(gt["propagation_affected_nodes"])
        top1_agent = ranked_agents[0]

        unaffected = gt["unaffected_nodes"]
        if unaffected:
            aff_idx = [agent_names.index(a) for a in affected]
            una_idx = [agent_names.index(a) for a in unaffected]
            score_ratio = float(scores[aff_idx].mean() / (scores[una_idx].mean() + 1e-8))
        else:
            score_ratio = None

        per_session.append({
            "session_id": gt["session_id"],
            "entry_rank": entry_rank,
            "entry_top1": entry_rank == 1,
            "entry_reciprocal_rank": 1.0 / entry_rank,
            "affected_hit_at_1": top1_agent in affected,
            "score_ratio": score_ratio,
        })
    return per_session


def main():
    templates_by_type = load_chain_templates()
    with open(META_ATTACK_PATH, encoding="utf-8") as f:
        meta_attack = json.load(f)
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    prop_gt = build_propagation_ground_truth(meta_attack, templates_by_type)
    gt_path = os.path.join(OUT, "propagation_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(prop_gt, f, indent=2)
    print(f"[saved] {gt_path}  ({len(prop_gt)} attack sessions)")

    n_no_unaffected = sum(1 for g in prop_gt if not g["unaffected_nodes"])
    print(f"  entry nodes used: {sorted({g['injection_entry_node'] for g in prop_gt})}  "
          f"(single entry point across all {len(prop_gt)} sessions in this v1 chain-only dataset)")
    print(f"  sessions with zero unaffected nodes: {n_no_unaffected}/{len(prop_gt)}  "
          f"(score_ratio undefined for these -- see module docstring)")

    per_seed_metrics = []
    for sd in results["per_seed"]:
        n_test_normal = sd["split_sizes"]["normal_test"]
        node_scores = sd["methods"]["LightGAE"]["test_node_scores"]
        attack_node_scores = node_scores[n_test_normal:]
        assert len(attack_node_scores) == len(prop_gt), \
            "attack rows in test_node_scores must align 1:1 with propagation_ground_truth order"
        per_session = node_localization_metrics(attack_node_scores, prop_gt)
        per_seed_metrics.append({
            "seed": sd["seed"],
            "entry_top1_accuracy": float(np.mean([p["entry_top1"] for p in per_session])),
            "entry_mrr": float(np.mean([p["entry_reciprocal_rank"] for p in per_session])),
            "affected_hit_at_1": float(np.mean([p["affected_hit_at_1"] for p in per_session])),
            "compromised_node_mean_rank": float(np.mean([p["entry_rank"] for p in per_session])),
            "score_ratio_mean": (float(np.mean([p["score_ratio"] for p in per_session if p["score_ratio"] is not None]))
                                  if any(p["score_ratio"] is not None for p in per_session) else None),
        })

    print(f"\n  {'seed':>6} {'entry_top1':>12} {'entry_MRR':>11} {'affected_hit@1':>15} {'mean_rank':>10}")
    for m in per_seed_metrics:
        print(f"  {m['seed']:>6} {m['entry_top1_accuracy']:>12.4f} {m['entry_mrr']:>11.4f} "
              f"{m['affected_hit_at_1']:>15.4f} {m['compromised_node_mean_rank']:>10.4f}")

    agg = {
        "entry_top1_accuracy_mean": float(np.mean([m["entry_top1_accuracy"] for m in per_seed_metrics])),
        "entry_mrr_mean": float(np.mean([m["entry_mrr"] for m in per_seed_metrics])),
        "affected_hit_at_1_mean": float(np.mean([m["affected_hit_at_1"] for m in per_seed_metrics])),
        "compromised_node_mean_rank_mean": float(np.mean([m["compromised_node_mean_rank"] for m in per_seed_metrics])),
        "score_ratio_mean": None,   # undefined dataset-wide -- see caveat below
    }
    print(f"\n  [mean across {len(per_seed_metrics)} seeds] entry_top1={agg['entry_top1_accuracy_mean']:.4f}  "
          f"entry_MRR={agg['entry_mrr_mean']:.4f}  affected_hit@1={agg['affected_hit_at_1_mean']:.4f}  "
          f"mean_rank={agg['compromised_node_mean_rank_mean']:.4f}")
    print("\n  [caveat -- affected_hit@1 / score_ratio] Trivial on this dataset, not a real")
    print("  benchmark result: every attack session's expected_propagation_path is all 4")
    print("  agents, so the affected set covers the whole graph (affected_hit@1=1.0 by")
    print("  construction) and there is no unaffected-node baseline for score_ratio.")
    print("  A meaningful reading of these two needs attack templates with a partial")
    print("  propagation path / varied target_agent (configs/attacks/direct.json,")
    print("  slow.json, length_preserving.json define this -- not yet wired into collection).")
    print("\n  [finding -- entry_top1 / MRR / mean_rank] NOT degenerate -- this IS a genuine")
    print("  1-of-4 prediction per session, and the result is informative: the injection")
    print("  entry node (Agent_0) is picked top-1 only ~2% of the time and has the LOWEST")
    print("  mean anomaly score of the 4 agents (see results_summary.json localization")
    print("  block: Agent_0=7.15 < Agent_1=10.31 < Agent_3=18.25 < Agent_2=29.48). LightGAE's")
    print("  node score currently tracks where the cascade effect is LARGEST (downstream,")
    print("  peaking at Agent_2), not where the attacker actually entered (Agent_0) -- i.e.")
    print("  it is a triage/impact signal, not an attribution signal. Matches the intended")
    print("  security-ops framing: node score prioritizes incident investigation, it does")
    print("  not confirm root cause.")

    out_path = os.path.join(OUT, "localization_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "purpose": "15 step -- propagation ground-truth schema + node-level localization "
                       "metrics, joined from configs/attacks/chain.json + session_metadata_attack.json "
                       "+ results_summary.json per_seed test_node_scores (no retraining, no new Ollama calls)",
            "propagation_ground_truth_file": gt_path,
            "caveat": "affected_hit_at_1 and score_ratio are trivial/undefined on the current v1 "
                      "chain-only dataset (expected_propagation_path = all 4 agents for every "
                      "session, so there is no unaffected-node baseline). entry_top1/entry_mrr/"
                      "compromised_node_mean_rank are NOT trivial and ARE informative: the entry "
                      "node is top-1 only ~2% of the time and has the lowest mean anomaly score of "
                      "the 4 agents -- LightGAE's node score currently tracks downstream cascade "
                      "impact, not the injection entry point (triage signal, not attribution). See "
                      "this file's docstring / stdout for detail.",
            "per_seed": per_seed_metrics,
            "aggregate": agg,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
