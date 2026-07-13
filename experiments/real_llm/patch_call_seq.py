"""
Offline patch: recompute call_seq (feature index 4) in the cached real-LLM
session data using the new joint token+ctx_delta definition, without
re-running Ollama. Requires cache_normal.json / cache_attack.json to already
contain token_count (index 1) and ctx_delta (index 3) from the original v4
collection run.

Old definition: call_seq = 1 if tokens > 280 else 0
  -> redundant with token_count (near-duplicate feature)
New definition: call_seq = 1 if (tokens > 280 and ctx_delta > 1.3) else 0
  -> joint deviation flag, consistent with the simulation-side fix
"""
import json
import os

OUT = "./output/real_llm"
FEAT_NAMES = ["latency", "token_count", "api_freq", "ctx_delta", "call_seq", "refusal_flag"]
TOK_IDX, CTX_IDX, SEQ_IDX = 1, 3, 4


def patch(path):
    with open(path) as f:
        data = json.load(f)
    changed = 0
    for session in data:
        for agent_feats in session:
            tokens    = agent_feats[TOK_IDX]
            ctx_delta = agent_feats[CTX_IDX]
            new_val   = 1.0 if (tokens > 280 and ctx_delta > 1.3) else 0.0
            if agent_feats[SEQ_IDX] != new_val:
                changed += 1
            agent_feats[SEQ_IDX] = new_val
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  {path}: {changed} call_seq values changed")


if __name__ == "__main__":
    for name in ["cache_normal.json", "cache_attack.json"]:
        patch(os.path.join(OUT, name))
    print("done.")
