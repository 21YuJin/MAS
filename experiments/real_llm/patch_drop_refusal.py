"""
One-off cache migration: drop the refusal_flag column (feature index 5) from
cached real-LLM session data, matching the 6->5 feature reduction in
lgnn_experiment.py (api_freq->sentence_count rename, refusal_flag removed).

Backups of the pre-migration 6-feature cache are kept as *.bak_6feat.
"""
import json
import os

OUT = "./output/real_llm"
DROP_IDX = 5  # refusal_flag


def patch(path):
    with open(path) as f:
        data = json.load(f)
    for session in data:
        for agent_feats in session:
            assert len(agent_feats) == 6, f"expected 6 features, got {len(agent_feats)}"
            del agent_feats[DROP_IDX]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  {path}: {len(data)} sessions, now {len(data[0][0])} features/agent")


if __name__ == "__main__":
    for name in ["cache_normal.json", "cache_attack.json"]:
        patch(os.path.join(OUT, name))
    print("done.")
