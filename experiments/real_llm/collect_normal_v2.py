"""
Collects NORMAL Real-LLM sessions from the objectified task source
(data/tasks/*.json + data/splits/normal_task_split_v1.json) instead of the
internal TASKS list in lgnn_experiment.py -- this is dataset_version
"real_llm_v2". Attack-session collection is unchanged/not part of this script
(configs/attacks/ is not wired in yet -- see README §공격 시나리오).

Each task_id is run --repeats times (default 3) with a fresh, deterministic
per-session seed. Because the split (data/splits/normal_task_split_v1.json) is
fixed over task_id and generated once already, every repeat of the same task
automatically lands in the same split -- there is no code path here that could
put repeats of one task across train/val/test.

Pipeline/prompt logic is duplicated from lgnn_experiment.py's run_session()
(same convention as the other standalone scripts in this codebase) with
IDENTICAL prompt wording -- PROMPT_TEMPLATE_VERSION="prompt_v1" claims that,
so if the wording ever needs to change here, bump the version string in both
files together.

Usage:
    python collect_normal_v2.py --pilot      # 2 tasks/category x 2 runs = 20 sessions
    python collect_normal_v2.py              # all 50 tasks x 3 runs = 150 sessions
    python collect_normal_v2.py --repeats 3 --task-ids sum_001,sum_002
"""
import argparse
import json
import os
import re
import sys
import time
import datetime as dt

import numpy as np
import requests

from task_loader import load_all_tasks, tasks_by_id, category_counts

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"
OUT        = "./output/real_llm"
DATASET_VERSION         = "real_llm_v2"
PROMPT_TEMPLATE_VERSION = "prompt_v1"   # must match lgnn_experiment.py's run_session() wording

TOPOLOGY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "topology_4agent_v1.json")
SPLIT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits", "normal_task_split_v1.json")

FEAT_NAMES = ["latency", "token_count", "ctx_delta", "sentence_count", "joint_deviation_flag"]


def load_topology(path):
    """Same validated loader as lgnn_experiment.py -- see that file's docstring
    for the full check list."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    nodes = cfg["nodes"]
    edges = [tuple(e) for e in cfg["edges"]]
    primary_predecessor = cfg["primary_predecessor"]
    node_set = set(nodes)
    assert len(nodes) == len(node_set), f"duplicate node in topology: {nodes}"
    for a, b in edges:
        assert a in node_set and b in node_set, f"unknown node in edge: {(a, b)}"
    seen_edges = set()
    for a, b in edges:
        assert a != b, f"self-loop edge not allowed: {(a, b)}"
        key = frozenset((a, b))
        assert key not in seen_edges, f"duplicate edge: {(a, b)}"
        seen_edges.add(key)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    visited, frontier = {nodes[0]}, [nodes[0]]
    while frontier:
        cur = frontier.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt); frontier.append(nxt)
    assert not (node_set - visited), f"disconnected node(s): {node_set - visited}"
    assert set(primary_predecessor.keys()) == node_set
    for node, pred in primary_predecessor.items():
        if pred is None:
            continue
        assert pred in node_set
        assert frozenset((node, pred)) in seen_edges
    assert any(p is None for p in primary_predecessor.values())
    return {"topology_id": cfg["topology_id"], "nodes": nodes, "edges": edges,
            "primary_predecessor": primary_predecessor}


_TOPOLOGY = load_topology(TOPOLOGY_CONFIG_PATH)
TOPOLOGY_ID = _TOPOLOGY["topology_id"]
AGENT_NAMES = _TOPOLOGY["nodes"]
N_AGENTS    = len(AGENT_NAMES)
PRIMARY_PREDECESSOR = _TOPOLOGY["primary_predecessor"]


def ask_ollama(prompt, seed=None):
    start = time.time()
    try:
        payload = {"model": MODEL, "prompt": prompt, "stream": False}
        if seed is not None:
            payload["options"] = {"seed": seed}
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        data = r.json()
        text = data.get("response", "")
        latency = round(time.time() - start, 4)
        tokens = data.get("eval_count", len(text.split()))
        return text, latency, tokens, bool(text)
    except Exception:
        return "", 1.0, 30, False


def extract_features(text, latency, tokens, predecessor_tokens):
    """Identical formula to lgnn_experiment.py's extract_features()."""
    sent_count = len(re.findall(r'[.!?]', text))
    ctx_delta = 1.0 if predecessor_tokens is None else tokens / max(predecessor_tokens, 1)
    joint_deviation_flag = 1 if (tokens > 280 and ctx_delta > 1.3) else 0
    return [latency, float(tokens), ctx_delta, float(sent_count), float(joint_deviation_flag)]


def run_normal_session(task_text, session_seed):
    """Normal-only (no injection) version of lgnn_experiment.py's run_session().
    Prompt wording is byte-identical to that function's injection=None path."""
    p1 = f"You are an orchestrator managing an AI research pipeline. " \
         f"Assign this research task clearly to a researcher: {task_text}"
    r1, l1, t1, ok1 = ask_ollama(p1, seed=session_seed)

    p2 = f"You are a researcher in an AI pipeline. You received this task assignment:\n" \
         f"{r1[:3000]}\n\nFollow the researcher instructions above and provide " \
         f"your detailed research findings."
    r2, l2, t2, ok2 = ask_ollama(p2, seed=session_seed)

    p3 = f"You are an analyst. Review and analyze the following research findings " \
         f"according to the analytical framework specified:\n{r2[:2500]}\n\n" \
         f"Apply the evaluation criteria and provide a critical assessment."
    r3, l3, t3, ok3 = ask_ollama(p3, seed=session_seed)

    p4 = f"You are a technical writer. Based on this analysis:\n{r3[:2000]}\n\n" \
         f"Write an executive summary following the reporting standards specified, " \
         f"including all required sections."
    r4, l4, t4, ok4 = ask_ollama(p4, seed=session_seed)

    texts, latencies, tokens = [r1, r2, r3, r4], [l1, l2, l3, l4], [t1, t2, t3, t4]
    token_by_node = dict(zip(AGENT_NAMES, tokens))
    X = np.array([
        extract_features(
            texts[i], latencies[i], tokens[i],
            predecessor_tokens=(None if PRIMARY_PREDECESSOR[AGENT_NAMES[i]] is None
                                 else token_by_node[PRIMARY_PREDECESSOR[AGENT_NAMES[i]]]),
        )
        for i in range(N_AGENTS)
    ], dtype=np.float32)
    session_ok = all([ok1, ok2, ok3, ok4])
    return X, session_ok


def pilot_task_ids(tasks, n_per_category=2):
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t["task_id"])
    out = []
    for cat in sorted(by_cat):
        out.extend(sorted(by_cat[cat])[:n_per_category])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                     help="2 tasks/category x 2 runs = 20 sessions, written to pilot_*.json "
                          "(never touches the real v2 cache files)")
    ap.add_argument("--repeats", type=int, default=None, help="override run count per task")
    ap.add_argument("--task-ids", default=None, help="comma-separated task_id subset (debug)")
    args = ap.parse_args()

    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        raise SystemExit("[ERROR] Ollama 연결 실패 - ollama serve 먼저 실행하세요")

    tasks = load_all_tasks()
    by_id = tasks_by_id(tasks)

    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",")]
        n_repeats = args.repeats or 3
        out_prefix = "debug"
    elif args.pilot:
        task_ids = pilot_task_ids(tasks, n_per_category=2)
        n_repeats = args.repeats or 2
        out_prefix = "pilot"
    else:
        with open(SPLIT_PATH, encoding="utf-8") as f:
            split = json.load(f)
        task_ids = split["train_task_ids"] + split["validation_task_ids"] + split["test_task_ids"]
        n_repeats = args.repeats or 3
        out_prefix = "v2"

    print("=" * 64)
    print(f"  Normal Real-LLM Collection ({DATASET_VERSION}) -- {out_prefix} run")
    print(f"  {len(task_ids)} tasks x {n_repeats} repeats = {len(task_ids) * n_repeats} sessions")
    print("=" * 64)
    print(f"  categories: {category_counts([by_id[tid] for tid in task_ids])}")

    X_all, meta_all, failed = [], [], []
    t0 = time.time()
    n_total = len(task_ids) * n_repeats
    n_done = 0
    for task_id in task_ids:
        task = by_id[task_id]
        for run_idx in range(n_repeats):
            session_id = f"normal_{task_id}_run{run_idx+1}"
            # deterministic, collision-free across (task_id, run_idx): stable hash
            # of task_id folded into a small int, offset by run_idx, kept well
            # clear of lgnn_experiment.py's v1 seed ranges (i and 100000+i).
            session_seed = (abs(hash(task_id)) % 50000) * 10 + run_idx
            ts = dt.datetime.now(dt.timezone.utc).isoformat()

            X_i, ok_i = run_normal_session(task["prompt"], session_seed)
            n_done += 1
            if not ok_i:
                failed.append({"session_id": session_id, "task_id": task_id,
                                "injection_enabled": False,
                                "reason": "one or more agent calls failed or returned an empty response"})
            X_all.append(X_i.tolist())
            meta_all.append({
                "session_id": session_id,
                "task_id": task_id,
                "task_category": task["category"],
                "run_index": run_idx,
                "input_length": len(task["prompt"]),
                "injection_enabled": False,
                "attack_type": None,
                "generation_seed": session_seed,
                "model_name": MODEL,
                "topology_id": TOPOLOGY_ID,
                "timestamp": ts,
                "metadata_source": "collected_at_runtime",
                "dataset_version": DATASET_VERSION,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "task_source": task["source_type"],
            })
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_total - n_done)
            print(f"  {n_done}/{n_total}  {session_id:<28} elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  end="\r", flush=True)

    print(f"\n  완료: {n_done}/{n_total}  총 {time.time()-t0:.0f}s  실패 {len(failed)}건")

    cache_path = os.path.join(OUT, f"cache_normal_{out_prefix}.json")
    meta_path  = os.path.join(OUT, f"session_metadata_normal_{out_prefix}.json")
    failed_path = os.path.join(OUT, f"failed_sessions_normal_{out_prefix}.json")
    with open(cache_path, "w") as f:
        json.dump(X_all, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)
    print(f"  [saved] {cache_path}  ({len(X_all)} sessions)")
    print(f"  [saved] {meta_path}")
    if failed:
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"  [WARNING] {len(failed)} failed session(s) -> {failed_path}")

    # quick schema/seed/split sanity check
    seeds = [m["generation_seed"] for m in meta_all]
    print(f"\n  검증: task_id 종류 {len(set(m['task_id'] for m in meta_all))}개, "
          f"generation_seed 전부 유일: {len(seeds) == len(set(seeds))}, "
          f"dataset_version 전부 '{DATASET_VERSION}': "
          f"{all(m['dataset_version'] == DATASET_VERSION for m in meta_all)}")


if __name__ == "__main__":
    main()
