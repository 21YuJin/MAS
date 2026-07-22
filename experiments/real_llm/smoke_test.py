"""
Phase 1 Smoke Test (P5/10순위, analysis_plan.md) -- 5 tasks x 2 conditions x 1
execution = 10 sessions. Cheap integration test before the 80-session feature
screening (6순위/12순위): confirms the instruction/content channel (P2/3순위)
and raw telemetry schema (P3/4순위) work end-to-end with REAL Ollama calls, for
BOTH conditions -- the attack condition was never run live through the new
pipeline before this (P3/4순위's own live verification only exercised normal).

Not a real experiment: no train/val/test split, no model training, no claims
about detection performance. Only validates plumbing, per analysis_plan.md's
Phase 1 checklist:
  - external content channel  (attack sessions show the expected downstream
                                 token-count cascade, same sanity check
                                 lgnn_experiment.py's §4 already prints)
  - raw telemetry              (no unexpected missing/None fields)
  - attack fields               (attack_type/attack_goal/indicator_observed
                                 populated for attack, None for normal)
  - schema                     (every telemetry record has the full key set)
  - feature extraction          (feature_pool_v2.py runs cleanly on REAL data,
                                 not just the synthetic data it was unit-tested
                                 against in 5순위)
  - done_reason                (what Ollama actually reports per call)
  - 오류 처리                    (error_flag/session_ok correctly reflect any
                                 failed calls, nothing silently swallowed)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
import lgnn_experiment as m          # noqa: E402  (triggers v1 cache-hit run on import, expected)
import feature_pool_v2 as fp         # noqa: E402

OUT = "./output/real_llm"
SMOKE_TASK_IDX = [0, 1, 2, 3, 4]

EXPECTED_TELEMETRY_KEYS = {
    "session_id", "task_id", "task_category", "task_source", "condition",
    "attack_type", "attack_goal", "execution_repeat", "hardware_backend",
    "gpu_name", "ollama_version", "agent_id", "sender_ids",
    "receiver_ids", "predecessor_ids", "execution_order", "ok", "error_flag",
    "retry_count", "prompt_eval_count", "eval_count", "prompt_eval_duration",
    "eval_duration", "total_duration", "load_duration", "wall_clock_latency_ms",
    "start_timestamp", "end_timestamp", "model", "temperature", "top_p",
    "num_predict", "done_reason", "response_text",
}


def run_smoke_test():
    all_telemetry = []
    sessions = []   # (session_id, condition, ok, telemetry)
    t0 = time.time()

    # [Session provenance addendum] Same warm-up + detect pattern as
    # lgnn_experiment.py §4 -- computed once, passed to every session below.
    # smoke_test.py is schema-validation only regardless of backend, but every
    # record still self-reports which backend actually produced it.
    m.ask_ollama("Say OK.")
    hw = m.detect_hardware_backend()
    print(f"  hardware_backend={hw['hardware_backend']}  gpu_name={hw['gpu_name']}  "
          f"ollama_version={hw['ollama_version']}")
    if hw["hardware_backend"] != "gpu":
        print(f"  [NOTE] running on {hw['hardware_backend']} -- this smoke test's output is "
              f"schema-validation only and must NOT be treated as formal v2 data "
              f"(analysis_plan.md). Every record below is self-labeled accordingly.")

    for n, idx in enumerate(SMOKE_TASK_IDX):
        task = m.TASKS[idx]
        task_id = f"task_{idx:03d}"
        category = m.TASK_CATEGORIES[idx]

        session_id = f"smoke_normal_{idx:03d}"
        X, indicator, ok, telemetry = m.run_session(
            task, injection=None, session_seed=500000 + idx, session_id=session_id,
            task_id=task_id, task_category=category, task_source="internal_TASKS_list",
            execution_repeat=0, hardware_backend=hw["hardware_backend"],
            gpu_name=hw["gpu_name"], ollama_version=hw["ollama_version"])
        all_telemetry.extend(telemetry)
        sessions.append((session_id, "normal", ok, telemetry))
        elapsed = time.time() - t0
        print(f"  [{2*n+1}/10] {session_id}  ok={ok}  elapsed={elapsed:.0f}s", flush=True)

        inj_idx = idx % len(m.INJECTIONS)
        session_id = f"smoke_attack_{idx:03d}"
        X2, indicator2, ok2, telemetry2 = m.run_session(
            task, injection=m.INJECTIONS[inj_idx], session_seed=600000 + idx, session_id=session_id,
            task_id=task_id, task_category=category, task_source="internal_TASKS_list",
            attack_type=m.ATTACK_TYPES[inj_idx],
            attack_goal="verbosity_inflation_v1_pending_redesign", execution_repeat=0,
            hardware_backend=hw["hardware_backend"], gpu_name=hw["gpu_name"],
            ollama_version=hw["ollama_version"])
        all_telemetry.extend(telemetry2)
        sessions.append((session_id, "attack", ok2, telemetry2))
        elapsed = time.time() - t0
        print(f"  [{2*n+2}/10] {session_id}  ok={ok2}  elapsed={elapsed:.0f}s", flush=True)

    raw_path = os.path.join(OUT, "smoke_test_raw_telemetry.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_telemetry, f, indent=2, ensure_ascii=False)
    print(f"\n  [saved] {raw_path}  ({len(all_telemetry)} records, "
          f"hardware_backend={hw['hardware_backend']}, schema-validation only)")

    return all_telemetry, sessions


def check_schema(all_telemetry):
    print("\n" + "=" * 64)
    print("  CHECK 1: schema (every record has the full expected key set)")
    print("=" * 64)
    bad = [r["session_id"] + "/" + r["agent_id"] for r in all_telemetry
           if set(r.keys()) != EXPECTED_TELEMETRY_KEYS]
    if bad:
        print(f"  [FAIL] {len(bad)} record(s) with unexpected key set: {bad}")
    else:
        print(f"  [OK] all {len(all_telemetry)} records match the expected schema")


def check_attack_fields(all_telemetry):
    print("\n" + "=" * 64)
    print("  CHECK 2: attack fields (populated for attack, None for normal)")
    print("=" * 64)
    bad = []
    for r in all_telemetry:
        is_attack = r["condition"] == "attack"
        has_attack_fields = r["attack_type"] is not None and r["attack_goal"] is not None
        if is_attack != has_attack_fields:
            bad.append(r["session_id"])
    if bad:
        print(f"  [FAIL] {len(bad)} session(s) with condition/attack_type mismatch: {bad}")
    else:
        n_attack = sum(1 for r in all_telemetry if r["condition"] == "attack")
        print(f"  [OK] attack_type/attack_goal populated for exactly the "
              f"{n_attack} attack-condition records, None elsewhere")


def check_raw_telemetry_completeness(all_telemetry):
    print("\n" + "=" * 64)
    print("  CHECK 3: raw telemetry (no unexpected None on successful calls)")
    print("=" * 64)
    required_when_ok = ["prompt_eval_count", "eval_count", "prompt_eval_duration",
                         "eval_duration", "total_duration", "load_duration", "done_reason"]
    bad = []
    for r in all_telemetry:
        if r["ok"]:
            missing = [k for k in required_when_ok if r.get(k) is None]
            if missing:
                bad.append((r["session_id"], r["agent_id"], missing))
    if bad:
        print(f"  [FAIL] {len(bad)} successful call(s) with missing fields: {bad}")
    else:
        print(f"  [OK] all successful calls have prompt_eval_count/eval_count/"
              f"durations/done_reason populated")

    done_reasons = {}
    for r in all_telemetry:
        done_reasons[r["done_reason"]] = done_reasons.get(r["done_reason"], 0) + 1
    print(f"  done_reason distribution: {done_reasons}")


def check_error_handling(all_telemetry, sessions):
    print("\n" + "=" * 64)
    print("  CHECK 4: 오류 처리 (error_flag/session_ok correctness)")
    print("=" * 64)
    n_errors = sum(1 for r in all_telemetry if r["error_flag"])
    n_not_ok = sum(1 for r in all_telemetry if not r["ok"])
    print(f"  error_flag=True: {n_errors}/{len(all_telemetry)}   ok=False: {n_not_ok}/{len(all_telemetry)}")
    session_fail = [(sid, cond) for sid, cond, ok, _ in sessions if not ok]
    if session_fail:
        print(f"  [WARNING] {len(session_fail)} session(s) had a failed/empty agent call: {session_fail}")
    else:
        print(f"  [OK] all {len(sessions)} sessions completed with every agent call successful")


def check_external_content_cascade(sessions):
    print("\n" + "=" * 64)
    print("  CHECK 5: external content channel (attack sessions show the expected")
    print("           downstream token-count cascade -- same sanity check as §4)")
    print("=" * 64)
    print(f"  {'task_idx':<10}{'Agent_0':>10}{'Agent_1':>10}{'Agent_2':>10}{'Agent_3':>10}   condition")
    for session_id, condition, ok, telemetry in sessions:
        by_order = sorted(telemetry, key=lambda r: r["execution_order"])
        toks = [r["eval_count"] for r in by_order]
        print(f"  {session_id:<20}{toks[0]:>10}{toks[1]:>10}{toks[2]:>10}{toks[3]:>10}   {condition}")


def check_feature_extraction(sessions):
    print("\n" + "=" * 64)
    print("  CHECK 6: feature extraction (feature_pool_v2.py on REAL data)")
    print("=" * 64)
    # Baseline fit on the 5 normal-condition smoke sessions -- NOT a real
    # train/val split (too small to mean anything statistically), just enough
    # to confirm fit_agent_zscore_baseline()/agent_zscore_features() don't
    # throw on real telemetry.
    normal_pairs = []
    for session_id, condition, ok, telemetry in sessions:
        if condition != "normal":
            continue
        by_id = {r["agent_id"]: r for r in telemetry}
        for r in telemetry:
            sender_ids = r.get("sender_ids") or []
            pred = by_id.get(sender_ids[0]) if sender_ids else None
            computed = fp.token_scale_features(r, predecessor_record=pred)
            computed.update(fp.timing_features(r))
            normal_pairs.append((r, computed))
    baseline = fp.fit_agent_zscore_baseline(normal_pairs)

    n_ok, n_fail = 0, 0
    sample = None
    for session_id, condition, ok, telemetry in sessions:
        try:
            node_feats, session_feats = fp.compute_session_feature_pool(telemetry, zscore_baseline=baseline)
            n_ok += 1
            if sample is None and condition == "attack":
                sample = (session_id, node_feats, session_feats)
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {session_id}: {type(e).__name__}: {e}")

    print(f"  [OK] compute_session_feature_pool() ran cleanly on {n_ok}/{len(sessions)} sessions"
          + (f", {n_fail} failure(s)" if n_fail else ""))
    if sample:
        sid, node_feats, session_feats = sample
        print(f"\n  sample (attack session {sid}, Agent_3 node features):")
        print(json.dumps(node_feats[3], indent=2))
        print(f"\n  sample (attack session {sid}, session-level features):")
        print(json.dumps(session_feats, indent=2))


def check_runtime_report(all_telemetry, sessions):
    """
    [GPU validation priority 1] Averages over successful calls only (error_flag
    records have no real duration and would silently drag the mean down) --
    plus extrapolated estimated runtime for 10/100/300-session collection
    sizes, extrapolated from THIS run's actual mean per-session wall-clock
    time (sum of its 4 agents' wall_clock_latency_ms), not a per-call figure
    multiplied blindly by 4 (agents differ systematically in output length,
    see the cascade check above, so a flat multiply would be biased).
    """
    print("\n" + "=" * 64)
    print("  RUNTIME REPORT (GPU validation priority 1)")
    print("=" * 64)

    ok_records = [r for r in all_telemetry if r["ok"]]
    if not ok_records:
        print("  [FAIL] no successful calls to report on")
        return

    def mean(key, records=ok_records):
        return sum(r[key] for r in records) / len(records)

    avg_eval_duration_ms = mean("eval_duration") / 1e6
    avg_prompt_eval_count = mean("prompt_eval_count")
    avg_eval_count = mean("eval_count")
    avg_wall_clock_ms = mean("wall_clock_latency_ms")
    tps_records = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in ok_records if r["eval_duration"]]
    avg_tokens_per_second = sum(tps_records) / len(tps_records) if tps_records else None

    print(f"  Average eval_duration:     {avg_eval_duration_ms:.1f} ms")
    print(f"  Average prompt_eval_count: {avg_prompt_eval_count:.1f} tokens")
    print(f"  Average eval_count:        {avg_eval_count:.1f} tokens")
    print(f"  Average tokens/sec:        {avg_tokens_per_second:.2f}" if avg_tokens_per_second
          else "  Average tokens/sec:        n/a")
    print(f"  Average wall_clock_latency: {avg_wall_clock_ms:.1f} ms")

    backends = {r["hardware_backend"] for r in all_telemetry}
    print(f"  hardware_backend(s) in this run: {sorted(backends)}")

    # Per-session wall-clock = sum of its 4 agents' wall_clock_latency_ms
    # (sequential pipeline -- agents don't run in parallel).
    per_session_ms = []
    for _session_id, _condition, _ok, telemetry in sessions:
        per_session_ms.append(sum(r["wall_clock_latency_ms"] for r in telemetry))
    avg_session_s = (sum(per_session_ms) / len(per_session_ms)) / 1000.0

    print(f"\n  Average session runtime: {avg_session_s:.1f}s  (this {len(sessions)}-session run)")
    print("\n  Estimated runtime (extrapolated from this run's average):")
    for n in (10, 100, 300):
        total_s = avg_session_s * n
        print(f"    {n:>4} sessions: {total_s/60:.1f} min  ({total_s/3600:.2f} h)")


if __name__ == "__main__":
    print("=" * 64)
    print("  Phase 1 Smoke Test -- 5 tasks x 2 conditions = 10 sessions")
    print("=" * 64)
    all_telemetry, sessions = run_smoke_test()
    check_schema(all_telemetry)
    check_attack_fields(all_telemetry)
    check_raw_telemetry_completeness(all_telemetry)
    check_error_handling(all_telemetry, sessions)
    check_external_content_cascade(sessions)
    check_feature_extraction(sessions)
    check_runtime_report(all_telemetry, sessions)
    print("\n실험 완료 (smoke test).")
