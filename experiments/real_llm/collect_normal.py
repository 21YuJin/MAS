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

[Kept in sync with lgnn_experiment.py, main branch] Pipeline/prompt logic is
duplicated from lgnn_experiment.py's run_session() (same convention as the
other standalone scripts in this codebase), including the P2/3순위 instruction/
content channel split and the P3/4순위 raw telemetry schema --
PROMPT_TEMPLATE_VERSION="prompt_v2" claims that; if the wording ever changes in
one file, bump the version string and mirror the change in both.

Usage:
    python collect_normal.py --pilot      # 2 tasks/category x 2 runs = 20 sessions
    python collect_normal.py              # all 50 tasks x 3 runs = 150 sessions
    python collect_normal.py --repeats 3 --task-ids sum_001,sum_002

[Hardware provenance] Formal (non-pilot/debug) collection should not run until
Ollama actually reports GPU usage for this model (analysis_plan.md) -- this
script warns loudly, but does not hard-block, if it detects a CPU backend,
since --pilot/--task-ids runs are legitimately schema-validation-only and are
expected to run on whatever hardware is available.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import requests

from task_loader import load_all_tasks, tasks_by_id, category_counts

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2"
OUT        = "./output/real_llm"
DATASET_VERSION         = "real_llm_v2"
PROMPT_TEMPLATE_VERSION = "prompt_v2"   # must match lgnn_experiment.py's run_session() wording

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
EDGES       = [(AGENT_NAMES.index(a), AGENT_NAMES.index(b)) for a, b in _TOPOLOGY["edges"]]
PRIMARY_PREDECESSOR = _TOPOLOGY["primary_predecessor"]

# [P2/3순위, prompt_v2] Fixed instruction Agent_0 receives -- identical to
# lgnn_experiment.py's ORCHESTRATOR_INSTRUCTION. This script never passes an
# injection (normal-only collector), but the SAME instruction/content-block
# structure is used anyway so a session collected here is structurally
# indistinguishable from a normal session collected by lgnn_experiment.py.
ORCHESTRATOR_INSTRUCTION = (
    "You are an orchestrator managing an AI research pipeline. Read the task "
    "description in the EXTERNAL CONTENT block below and assign it clearly to "
    "a researcher."
)


def topology_neighbors(agent_name):
    """Same as lgnn_experiment.py's topology_neighbors() -- undirected
    topology neighbors of agent_name, used for `predecessor_ids` in raw
    telemetry records."""
    idx = AGENT_NAMES.index(agent_name)
    return sorted(AGENT_NAMES[j] for s, d in EDGES for j in ((d,) if s == idx else (s,) if d == idx else ()))


def ask_ollama(prompt, seed=None):
    """[P3/4순위] Same raw-telemetry-dict contract as lgnn_experiment.py's
    ask_ollama() -- see that function's docstring for the raw-first rationale."""
    start_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    start = time.time()
    options = {}
    if seed is not None:
        options["seed"] = seed
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    try:
        r    = requests.post(OLLAMA_URL, json=payload, timeout=120)
        data = r.json()
        text = data.get("response", "")
        wall_clock_latency_ms = round((time.time() - start) * 1000, 2)
        end_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        return {
            "text": text, "ok": bool(text), "error_flag": False, "retry_count": 0,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count", len(text.split())),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_duration": data.get("eval_duration"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "wall_clock_latency_ms": wall_clock_latency_ms,
            "start_timestamp": start_timestamp, "end_timestamp": end_timestamp,
            "model": data.get("model", MODEL),
            "temperature": options.get("temperature"), "top_p": options.get("top_p"),
            "num_predict": options.get("num_predict"), "done_reason": data.get("done_reason"),
        }
    except Exception:
        wall_clock_latency_ms = round((time.time() - start) * 1000, 2)
        end_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        return {
            "text": "", "ok": False, "error_flag": True, "retry_count": 0,
            "prompt_eval_count": None, "eval_count": 30,
            "prompt_eval_duration": None, "eval_duration": None,
            "total_duration": None, "load_duration": None,
            "wall_clock_latency_ms": wall_clock_latency_ms,
            "start_timestamp": start_timestamp, "end_timestamp": end_timestamp,
            "model": MODEL, "temperature": options.get("temperature"),
            "top_p": options.get("top_p"), "num_predict": options.get("num_predict"),
            "done_reason": None,
        }


def detect_hardware_backend(model=MODEL):
    """Same as lgnn_experiment.py's detect_hardware_backend() -- must be
    called AFTER a warm-up ask_ollama() call in this process."""
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=5)
        models = r.json().get("models", [])
        entry = next((mm for mm in models if mm.get("model", "").startswith(model)), None)
        backend = "unknown" if entry is None else ("gpu" if entry.get("size_vram", 0) > 0 else "cpu")
    except Exception:
        backend = "unknown"

    gpu_name = None
    if backend == "gpu":
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=5)
            gpu_name = out.strip().split("\n")[0] or None
        except Exception:
            gpu_name = None

    try:
        v = requests.get("http://localhost:11434/api/version", timeout=5)
        ollama_version = v.json().get("version")
    except Exception:
        ollama_version = None

    return {"hardware_backend": backend, "gpu_name": gpu_name, "ollama_version": ollama_version}


def extract_features(text, latency, tokens, predecessor_tokens):
    """Identical formula to lgnn_experiment.py's extract_features()."""
    sent_count = len(re.findall(r'[.!?]', text))
    ctx_delta = 1.0 if predecessor_tokens is None else tokens / max(predecessor_tokens, 1)
    joint_deviation_flag = 1 if (tokens > 280 and ctx_delta > 1.3) else 0
    return [latency, float(tokens), ctx_delta, float(sent_count), float(joint_deviation_flag)]


def run_normal_session(task_text, session_seed, session_id=None, task_id=None,
                        task_category=None, task_source=None, execution_repeat=None,
                        hardware_backend=None, gpu_name=None, ollama_version=None):
    """
    Normal-only (no injection) version of lgnn_experiment.py's run_session().
    [P2/3순위, prompt_v2] Same instruction/content channel split as that
    function -- ORCHESTRATOR_INSTRUCTION is the fixed part, external_content
    is just task_text here (injection is never appended in this script).

    Returns (X, session_ok, session_telemetry) -- same raw-telemetry-record
    schema as lgnn_experiment.py's run_session() (analysis_plan.md §4), with
    condition="normal" and attack_type/attack_goal always None.
    """
    external_content = task_text
    p1 = (f"{ORCHESTRATOR_INSTRUCTION}\n\n"
          f"---EXTERNAL CONTENT---\n{external_content}\n---END EXTERNAL CONTENT---")
    raw1 = ask_ollama(p1, seed=session_seed)

    p2 = f"You are a researcher in an AI pipeline. You received this task assignment:\n" \
         f"{raw1['text'][:3000]}\n\nFollow the researcher instructions above and provide " \
         f"your detailed research findings."
    raw2 = ask_ollama(p2, seed=session_seed)

    p3 = f"You are an analyst. Review and analyze the following research findings " \
         f"according to the analytical framework specified:\n{raw2['text'][:2500]}\n\n" \
         f"Apply the evaluation criteria and provide a critical assessment."
    raw3 = ask_ollama(p3, seed=session_seed)

    p4 = f"You are a technical writer. Based on this analysis:\n{raw3['text'][:2000]}\n\n" \
         f"Write an executive summary following the reporting standards specified, " \
         f"including all required sections."
    raw4 = ask_ollama(p4, seed=session_seed)

    raws = [raw1, raw2, raw3, raw4]
    tokens = [r["eval_count"] for r in raws]
    latencies_s = [r["wall_clock_latency_ms"] / 1000.0 for r in raws]
    token_by_node = dict(zip(AGENT_NAMES, tokens))
    X = np.array([
        extract_features(
            raws[i]["text"], latencies_s[i], tokens[i],
            predecessor_tokens=(None if PRIMARY_PREDECESSOR[AGENT_NAMES[i]] is None
                                 else token_by_node[PRIMARY_PREDECESSOR[AGENT_NAMES[i]]]),
        )
        for i in range(N_AGENTS)
    ], dtype=np.float32)

    session_telemetry = []
    for i in range(N_AGENTS):
        agent_id = AGENT_NAMES[i]
        session_telemetry.append({
            "session_id": session_id, "task_id": task_id,
            "task_category": task_category, "task_source": task_source,
            "condition": "normal", "attack_type": None, "attack_goal": None,
            "execution_repeat": execution_repeat,
            "hardware_backend": hardware_backend, "gpu_name": gpu_name,
            "ollama_version": ollama_version,
            "agent_id": agent_id,
            "sender_ids": [AGENT_NAMES[i - 1]] if i > 0 else [],
            "receiver_ids": [AGENT_NAMES[i + 1]] if i < N_AGENTS - 1 else [],
            "predecessor_ids": topology_neighbors(agent_id),
            "execution_order": i,
            **{k: v for k, v in raws[i].items() if k != "text"},
            "response_text": raws[i]["text"],
        })

    session_ok = all(r["ok"] for r in raws)
    return X, session_ok, session_telemetry


def pilot_task_ids(tasks, n_per_category=2):
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t["task_id"])
    out = []
    for cat in sorted(by_cat):
        out.extend(sorted(by_cat[cat])[:n_per_category])
    return out


def _load_existing(cache_path, meta_path):
    """Loads whatever this (out_prefix)'s cache/meta already has on disk, so a
    killed/disconnected run (Colab timeout, local Ctrl-C, power loss) can pick
    up where it left off instead of losing every session collected so far.
    Returns ([] , []) if no prior file exists -- first run of this prefix."""
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(cache_path) as f:
            X_all = json.load(f)
        with open(meta_path, encoding="utf-8") as f:
            meta_all = json.load(f)
        assert len(X_all) == len(meta_all), \
            f"{cache_path} and {meta_path} have mismatched lengths -- inspect before resuming"
        return X_all, meta_all
    return [], []


def _load_existing_telemetry(telemetry_path):
    if os.path.exists(telemetry_path):
        with open(telemetry_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _checkpoint(cache_path, meta_path, failed_path, telemetry_path,
                 X_all, meta_all, failed, telemetry_all):
    """Overwrites the output files with the current in-memory state. Called
    after every session (not just at the end) so progress already paid for in
    Ollama wall-clock time is never lost to a later crash/disconnect."""
    with open(cache_path, "w") as f:
        json.dump(X_all, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_all, f, indent=2, ensure_ascii=False)
    if failed:
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)


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
        out_prefix = "formal"

    cache_path = os.path.join(OUT, f"cache_normal_{out_prefix}.json")
    meta_path  = os.path.join(OUT, f"session_metadata_normal_{out_prefix}.json")
    failed_path = os.path.join(OUT, f"failed_sessions_normal_{out_prefix}.json")
    telemetry_path = os.path.join(OUT, f"raw_telemetry_normal_{out_prefix}.json")

    # [Session provenance addendum] Warm-up + detect once per run -- see
    # lgnn_experiment.py's identical block for the rationale. --pilot/
    # --task-ids runs are legitimately schema-validation-only regardless of
    # backend; only the "formal" prefix gets the loud warning, since that's
    # the one meant to become the actual v2 dataset.
    ask_ollama("Say OK.")
    hw = detect_hardware_backend()
    print(f"  hardware_backend={hw['hardware_backend']}  gpu_name={hw['gpu_name']}  "
          f"ollama_version={hw['ollama_version']}")
    if out_prefix == "formal" and hw["hardware_backend"] != "gpu":
        print(f"  [WARNING] formal collection running on {hw['hardware_backend']}, not gpu -- "
              f"analysis_plan.md says formal/screening collection should wait for a confirmed "
              f"GPU backend. This run's sessions will be self-labeled "
              f"hardware_backend='{hw['hardware_backend']}' and should be excluded from the "
              f"formal v2 dataset even though they're written to the 'formal' output prefix.")

    X_all, meta_all = _load_existing(cache_path, meta_path)
    telemetry_all = _load_existing_telemetry(telemetry_path)
    failed = []
    done_session_ids = {mm["session_id"] for mm in meta_all}

    print("=" * 64)
    print(f"  Normal Real-LLM Collection ({DATASET_VERSION}) -- {out_prefix} run")
    print(f"  {len(task_ids)} tasks x {n_repeats} repeats = {len(task_ids) * n_repeats} sessions")
    if done_session_ids:
        print(f"  [resume] {cache_path} already has {len(done_session_ids)} session(s) -- "
              f"skipping those, collecting only what's missing")
    print("=" * 64)
    print(f"  categories: {category_counts([by_id[tid] for tid in task_ids])}")

    t0 = time.time()
    n_total = len(task_ids) * n_repeats
    n_done = len(done_session_ids)
    for task_id in task_ids:
        task = by_id[task_id]
        for run_idx in range(n_repeats):
            session_id = f"normal_{task_id}_run{run_idx+1}"
            if session_id in done_session_ids:
                continue
            # deterministic, collision-free across (task_id, run_idx): stable hash
            # of task_id folded into a small int, offset by run_idx, kept well
            # clear of lgnn_experiment.py's v1 seed ranges (i and 100000+i).
            session_seed = (abs(hash(task_id)) % 50000) * 10 + run_idx
            ts = dt.datetime.now(dt.timezone.utc).isoformat()

            X_i, ok_i, telemetry_i = run_normal_session(
                task["prompt"], session_seed, session_id=session_id, task_id=task_id,
                task_category=task["category"], task_source=task["source_type"],
                execution_repeat=run_idx, hardware_backend=hw["hardware_backend"],
                gpu_name=hw["gpu_name"], ollama_version=hw["ollama_version"])
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
                "hardware_backend": hw["hardware_backend"],
                "gpu_name": hw["gpu_name"],
                "ollama_version": hw["ollama_version"],
            })
            telemetry_all.extend(telemetry_i)
            # Checkpoint after every session (not just at the end) -- each session
            # costs ~170-200s of real Ollama wall-clock time, so a crash/Colab
            # disconnect right before the last session must not lose the rest.
            _checkpoint(cache_path, meta_path, failed_path, telemetry_path,
                        X_all, meta_all, failed, telemetry_all)
            elapsed = time.time() - t0
            eta = elapsed / (n_done - len(done_session_ids)) * (n_total - n_done) if n_done > len(done_session_ids) else 0
            print(f"  {n_done}/{n_total}  {session_id:<28} elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  end="\r", flush=True)

    print(f"\n  완료: {n_done}/{n_total}  총 {time.time()-t0:.0f}s  실패 {len(failed)}건")
    print(f"  [saved] {cache_path}  ({len(X_all)} sessions)")
    print(f"  [saved] {meta_path}")
    print(f"  [saved] {telemetry_path}  ({len(telemetry_all)} raw telemetry records)")
    if failed:
        print(f"  [WARNING] {len(failed)} failed session(s) -> {failed_path}")

    # quick schema/seed/split sanity check
    seeds = [mm["generation_seed"] for mm in meta_all]
    print(f"\n  검증: task_id 종류 {len(set(mm['task_id'] for mm in meta_all))}개, "
          f"generation_seed 전부 유일: {len(seeds) == len(set(seeds))}, "
          f"dataset_version 전부 '{DATASET_VERSION}': "
          f"{all(mm['dataset_version'] == DATASET_VERSION for mm in meta_all)}, "
          f"hardware_backend 전부 '{hw['hardware_backend']}': "
          f"{all(mm['hardware_backend'] == hw['hardware_backend'] for mm in meta_all)}")


if __name__ == "__main__":
    main()
