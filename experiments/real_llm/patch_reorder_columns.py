"""
One-off migration: swap cached feature columns 2 and 3 (sentence_count <-> ctx_delta)
to match the new FEAT_NAMES order:
  old: [latency, token_count, sentence_count, ctx_delta, joint_deviation_flag]
  new: [latency, token_count, ctx_delta, sentence_count, joint_deviation_flag]
Backups (.bak_pre_reorder) were taken before running this.
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "real_llm")
CACHE_NORMAL = os.path.join(OUT, "cache_normal.json")
CACHE_ATTACK = os.path.join(OUT, "cache_attack.json")

def swap_cols(path):
    with open(path) as f:
        data = json.load(f)
    for session in data:
        for agent_row in session:
            agent_row[2], agent_row[3] = agent_row[3], agent_row[2]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  patched {path} ({len(data)} sessions)")

swap_cols(CACHE_NORMAL)
swap_cols(CACHE_ATTACK)
print("done")
