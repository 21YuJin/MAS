"""
Candidate feature pool (P4/5순위, analysis_plan.md §5) computed from raw
telemetry -- the session_telemetry records lgnn_experiment.py's run_session()
now returns (P3/4순위). Deliberately NOT narrowed to a fixed Core set yet:
screening (6순위, 20-task normal/attack data) decides what survives, using
the exclusion criteria in analysis_plan.md §6 (near-constant, high missing
rate, unstable definition, unavailable in deployment, directly encodes the
label, |r|>0.95 redundant with another feature).

Every function here is a pure function of fields already stored in raw
telemetry -- none of them require re-running Ollama, so changing which
features survive screening never forces recollection. This mirrors v1's
CORE_FEATURES/DIAGNOSTIC_FEATURES split (lgnn_experiment.py) one level
earlier: raw telemetry -> (this file) candidate pool -> (6~7순위) screened
Core set -> model input.

Five categories (§5 A-E):
  A. token-scale       -- per node, single record, highest priority
  B. timing/speed      -- per node, single record
  C. agent-normalized  -- per node, needs a baseline FIT ON NORMAL-TRAIN ONLY
  D. session-level      -- one value per session (NOT per node -- kept
                           separate, not replicated across the 4 agents)
  E. orchestration      -- per node, from session structure/topology
"""
from collections import defaultdict

import numpy as np

NS_TO_MS = 1e-6


def _get(record, key, default=0):
    v = record.get(key)
    return default if v is None else v


# ══════════════════════════════════════════════════════════════════════════
# A. Token-scale features (§5.A) -- highest priority
# ══════════════════════════════════════════════════════════════════════════

def token_scale_features(record, predecessor_record=None):
    """
    record: one session_telemetry entry (a single agent call).
    predecessor_record: the PRIMARY_PREDECESSOR's record in the same session,
    or None for an entry node (no incoming primary predecessor -- matches v1's
    ctx_delta_entry=1.0 convention: predecessor_output_ratio=1.0,
    predecessor_output_difference=0.0 for entry nodes, not NaN/undefined).

    ctx_delta (v1) is NOT deleted -- it is exactly predecessor_output_ratio
    below, renamed per analysis_plan.md §5.A ("기존 ctx_delta는 삭제하지 말고
    predecessor_output_ratio로 이름을 바꿔 보조 후보로 남겨").
    """
    input_tokens  = _get(record, "prompt_eval_count")
    output_tokens = _get(record, "eval_count")
    total_tokens  = input_tokens + output_tokens
    expansion_ratio = output_tokens / max(input_tokens, 1)
    input_output_difference = output_tokens - input_tokens

    if predecessor_record is not None:
        pred_output = _get(predecessor_record, "eval_count")
        predecessor_output_ratio = output_tokens / max(pred_output, 1)
        predecessor_output_difference = output_tokens - pred_output
    else:
        predecessor_output_ratio = 1.0
        predecessor_output_difference = 0.0

    return {
        "input_token_count": float(input_tokens),
        "output_token_count": float(output_tokens),
        "total_token_count": float(total_tokens),
        "expansion_ratio": float(expansion_ratio),
        "input_output_difference": float(input_output_difference),
        "predecessor_output_ratio": float(predecessor_output_ratio),
        "predecessor_output_difference": float(predecessor_output_difference),
    }


# ══════════════════════════════════════════════════════════════════════════
# B. Timing / speed features (§5.B)
# ══════════════════════════════════════════════════════════════════════════

def timing_features(record):
    """
    All *_ms fields are converted from Ollama's raw nanosecond durations.
    load_duration_ms is included (not part of the named list in §5.B, but the
    section explicitly requires storing it "반드시 같이" for warm-up/GPU-state
    diagnosis) -- kept as a diagnostic companion to generation_time_ratio, not
    a candidate feature to screen on its own merit.
    """
    prompt_eval_time_ms = _get(record, "prompt_eval_duration") * NS_TO_MS
    generation_time_ms  = _get(record, "eval_duration") * NS_TO_MS
    total_duration_ms   = _get(record, "total_duration") * NS_TO_MS
    load_duration_ms    = _get(record, "load_duration") * NS_TO_MS
    wall_clock_latency_ms = _get(record, "wall_clock_latency_ms")

    output_tokens = _get(record, "eval_count")
    input_tokens  = _get(record, "prompt_eval_count")
    generation_time_s   = generation_time_ms / 1000.0
    prompt_eval_time_s  = prompt_eval_time_ms / 1000.0

    tokens_per_second = (output_tokens / generation_time_s) if generation_time_s > 0 else None
    prompt_tokens_per_second = (input_tokens / prompt_eval_time_s) if prompt_eval_time_s > 0 else None
    non_generation_overhead_ms = wall_clock_latency_ms - generation_time_ms
    generation_time_ratio = generation_time_ms / max(total_duration_ms, 1e-6)

    return {
        "prompt_eval_time_ms": float(prompt_eval_time_ms),
        "generation_time_ms": float(generation_time_ms),
        "total_duration_ms": float(total_duration_ms),
        "wall_clock_latency_ms": float(wall_clock_latency_ms),
        "tokens_per_second": (float(tokens_per_second) if tokens_per_second is not None else None),
        "prompt_tokens_per_second": (float(prompt_tokens_per_second) if prompt_tokens_per_second is not None else None),
        "non_generation_overhead_ms": float(non_generation_overhead_ms),
        "generation_time_ratio": float(generation_time_ratio),
        "load_duration_ms": float(load_duration_ms),   # diagnostic companion, see docstring
    }


# ══════════════════════════════════════════════════════════════════════════
# C. Agent-normalized (z-score) features (§5.C) -- fit on normal-TRAIN only
# ══════════════════════════════════════════════════════════════════════════

# Which A/B features get a z-score companion, and the §5.C output name for each.
ZSCORE_SOURCE_FEATURES = {
    "input_token_count": "agent_input_zscore",
    "output_token_count": "agent_output_zscore",
    "expansion_ratio": "agent_expansion_zscore",
    "generation_time_ms": "agent_generation_time_zscore",
    "tokens_per_second": "agent_tokens_per_second_zscore",
}


def fit_agent_zscore_baseline(normal_train_records_with_features):
    """
    normal_train_records_with_features: list of (record, computed) pairs --
    `record` is the raw session_telemetry entry (only used for agent_id here),
    `computed` is token_scale_features(record, pred) | timing_features(record)
    already merged for that same record. MUST be normal-train sessions only --
    this is enforced by the caller passing that subset in, not by this
    function (which has no way to check "is this normal" on its own; the split
    itself is external, same as StandardScaler.fit() in lgnn_experiment.py).

    Returns {agent_id: {source_feature_name: (mean, std)}}. tokens_per_second
    can be None for a record with zero generation_time -- such records are
    skipped for that feature's mean/std, not treated as 0.
    """
    by_agent = defaultdict(lambda: defaultdict(list))
    for record, computed in normal_train_records_with_features:
        agent_id = record["agent_id"]
        for feat in ZSCORE_SOURCE_FEATURES:
            val = computed.get(feat)
            if val is not None:
                by_agent[agent_id][feat].append(val)

    baseline = {}
    for agent_id, feats in by_agent.items():
        baseline[agent_id] = {}
        for feat, values in feats.items():
            arr = np.asarray(values, dtype=float)
            baseline[agent_id][feat] = (float(arr.mean()), float(arr.std()))
    return baseline


def agent_zscore_features(record, computed, baseline):
    """
    computed: token_scale_features(record, pred) | timing_features(record) for
    THIS record (any session, normal or attack -- only the baseline itself is
    normal-train-only). Returns None for a feature if this agent_id has no
    baseline entry (e.g. every normal-train value for that feature was None)
    rather than fabricating a 0.
    """
    agent_id = record["agent_id"]
    agent_baseline = baseline.get(agent_id, {})
    out = {}
    for feat, out_name in ZSCORE_SOURCE_FEATURES.items():
        val = computed.get(feat)
        stats = agent_baseline.get(feat)
        if val is None or stats is None:
            out[out_name] = None
            continue
        mean, std = stats
        out[out_name] = float((val - mean) / (std + 1e-8))
    return out


# ══════════════════════════════════════════════════════════════════════════
# D. Session-level features (§5.D) -- one value per SESSION, not per node
# ══════════════════════════════════════════════════════════════════════════

def session_level_features(session_records):
    """
    session_records: the N_AGENTS raw telemetry records for ONE session (any
    order -- sorted by execution_order internally). Returns a single flat
    dict, meant to be stored/joined at the session level (graph-level
    baseline / hybrid readout per §5.D), never copied into each node's own
    feature vector.

    output_concentration: share of total session output produced by the single
    highest-output node (max/sum) -- a concentration measure in [1/N, 1], where
    1/N means output is spread evenly and 1 means one node produced everything.
    Not explicitly formularized in analysis_plan.md §5.D; this is this file's
    own definition, documented rather than left implicit.
    """
    records = sorted(session_records, key=lambda r: r["execution_order"])
    inputs    = np.array([_get(r, "prompt_eval_count") for r in records], dtype=float)
    outputs   = np.array([_get(r, "eval_count") for r in records], dtype=float)
    latencies = np.array([_get(r, "wall_clock_latency_ms") for r in records], dtype=float)

    def cv(arr):
        m = arr.mean()
        return float(arr.std() / m) if m > 0 else None

    return {
        "session_input_mean": float(inputs.mean()),
        "session_input_std": float(inputs.std()),
        "session_output_mean": float(outputs.mean()),
        "session_output_std": float(outputs.std()),
        "session_output_cv": cv(outputs),
        "session_latency_mean": float(latencies.mean()),
        "session_latency_std": float(latencies.std()),
        "session_latency_cv": cv(latencies),
        "max_output_token": float(outputs.max()),
        "min_output_token": float(outputs.min()),
        "max_min_output_ratio": float(outputs.max() / max(outputs.min(), 1)),
        "output_concentration": float(outputs.max() / max(outputs.sum(), 1)),
    }


# ══════════════════════════════════════════════════════════════════════════
# E. Structure / orchestration features (§5.E)
# ══════════════════════════════════════════════════════════════════════════

def orchestration_features(record, n_agents):
    """
    incoming/outgoing_message_count come from sender_ids/receiver_ids (actual
    linear prompt-chain flow, run_session()'s own bookkeeping). fan_in/fan_out
    come from predecessor_ids (topology_neighbors() -- undirected graph degree,
    can differ from message flow, see run_session()'s docstring). In THIS
    codebase's topology_4agent_v1 these are NOT constant across agents
    (Agent_0/1 have degree 2, Agent_2 has degree 3, Agent_3 has degree 1) --
    §5.E's "고정 topology에서 항상 같은 값이면 제거" caveat doesn't apply here,
    but is still checked per-topology by screening (6순위), not assumed.
    """
    sender_ids      = record.get("sender_ids") or []
    receiver_ids    = record.get("receiver_ids") or []
    predecessor_ids = record.get("predecessor_ids") or []
    execution_order = record["execution_order"]

    incoming = len(sender_ids)
    outgoing = len(receiver_ids)
    fan_in   = len(predecessor_ids)
    fan_out  = fan_in   # topology edges are undirected in this codebase (§EDGES) --
                         # degree is the same viewed from either side of every edge.

    return {
        "message_count": incoming + outgoing,
        "incoming_message_count": incoming,
        "outgoing_message_count": outgoing,
        "fan_in": fan_in,
        "fan_out": fan_out,
        "execution_order": execution_order,
        "node_position_ratio": execution_order / max(n_agents - 1, 1),
        "error_count": int(bool(record.get("error_flag"))),
        "retry_count": _get(record, "retry_count"),
    }


def session_timing_orchestration_features(session_records):
    """
    inter_agent_delay_ms / cumulative_elapsed_time_ms need the WHOLE session
    (not a single record) -- computed here as a separate per-node list,
    parallel to orchestration_features() above, rather than folded into it.
    inter_agent_delay_ms[i] = gap between agent i-1's end_timestamp and agent
    i's start_timestamp (queueing/dispatch overhead between agents, distinct
    from any single agent's own generation time) -- 0.0 for the entry node
    (i=0, nothing precedes it). cumulative_elapsed_time_ms[i] = wall-clock time
    from session start through agent i's completion.
    """
    import datetime as dt

    records = sorted(session_records, key=lambda r: r["execution_order"])

    def parse(ts):
        return dt.datetime.fromisoformat(ts)

    session_start = parse(records[0]["start_timestamp"])
    out = []
    prev_end = None
    for r in records:
        start = parse(r["start_timestamp"])
        end   = parse(r["end_timestamp"])
        inter_agent_delay_ms = (
            0.0 if prev_end is None else (start - prev_end).total_seconds() * 1000.0
        )
        cumulative_elapsed_time_ms = (end - session_start).total_seconds() * 1000.0
        out.append({
            "inter_agent_delay_ms": float(inter_agent_delay_ms),
            "cumulative_elapsed_time_ms": float(cumulative_elapsed_time_ms),
        })
        prev_end = end
    return out


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator: build the full candidate pool for one session
# ══════════════════════════════════════════════════════════════════════════

def compute_session_feature_pool(session_records, zscore_baseline=None):
    """
    session_records: N_AGENTS raw telemetry records for one session (any order).
    zscore_baseline: output of fit_agent_zscore_baseline() on normal-train data,
    or None to skip category C entirely (e.g. before a baseline exists yet).

    Returns (node_features, session_features):
      node_features: list of dicts, one per agent, execution_order-sorted,
                     each containing categories A + B + C(if baseline given) + E
      session_features: single dict, category D (§5.D -- NOT replicated per node)
    """
    records = sorted(session_records, key=lambda r: r["execution_order"])
    by_agent_id = {r["agent_id"]: r for r in records}

    # predecessor_record for category A: this session's PRIMARY_PREDECESSOR
    # relationship is encoded in run_session()'s predecessor_ids/topology, but
    # A's "predecessor" specifically means the ctx_delta-style single primary
    # predecessor -- reconstructed here the same way lgnn_experiment.py's
    # PRIMARY_PREDECESSOR does: the sender in the linear prompt chain, i.e.
    # sender_ids[0] if present (this pipeline's chain has at most one sender).
    orch_records = session_timing_orchestration_features(records)

    node_features = []
    for i, r in enumerate(records):
        sender_ids = r.get("sender_ids") or []
        pred_record = by_agent_id.get(sender_ids[0]) if sender_ids else None

        feats = {}
        feats.update(token_scale_features(r, predecessor_record=pred_record))
        computed_for_zscore = dict(feats)   # A's outputs feed C
        feats.update(timing_features(r))
        computed_for_zscore.update(timing_features(r))   # B's outputs also feed C
        if zscore_baseline is not None:
            feats.update(agent_zscore_features(r, computed_for_zscore, zscore_baseline))
        feats.update(orchestration_features(r, n_agents=len(records)))
        feats.update(orch_records[i])
        feats["agent_id"] = r["agent_id"]
        node_features.append(feats)

    session_features = session_level_features(records)
    return node_features, session_features
