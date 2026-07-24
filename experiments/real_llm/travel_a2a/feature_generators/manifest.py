"""
[Step 7B] Builds feature_generation_manifest.json by running the candidate
feature generator over all 50 formal-workload mock sessions (deterministic,
no Ollama calls). This is a DIAGNOSTIC reproducibility manifest, not feature
screening: it records a determinism hash and which candidate fields are
populated vs. structurally null under mock execution -- no correlation
filtering, no AUROC, no feature selection happens here.
"""
import copy
import hashlib
import json
import os
from typing import Any, Dict, List

from ..formal_workload_mock_run import run_all_formal_mock_sessions
from .generate import generate_candidate_features_for_session

DEFAULT_REPORT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "reports", "travel_a2a", "feature_pool"))


def _null_rate(rows: List[Dict[str, Any]], field: str) -> float:
    if not rows:
        return 1.0
    nulls = sum(1 for r in rows if r.get(field) is None)
    return nulls / len(rows)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def build_feature_generation_manifest() -> Dict[str, Any]:
    outcome_pairs = run_all_formal_mock_sessions(save_sessions=False)

    per_task: Dict[str, Any] = {}
    all_token_rows: List[Dict[str, Any]] = []
    all_timing_rows: List[Dict[str, Any]] = []
    for outcome, result in outcome_pairs:
        before = ([c.to_dict() for c in result.agent_call_records], [e.to_dict() for e in result.events],
                  [m.to_dict() for m in result.messages], [a.to_dict() for a in result.artifacts],
                  [p.to_dict() for p in result.parts])
        before_copy = copy.deepcopy(before)

        features = generate_candidate_features_for_session(
            result.agent_call_records, result.events, result.messages, result.artifacts, result.parts)

        after = ([c.to_dict() for c in result.agent_call_records], [e.to_dict() for e in result.events],
                 [m.to_dict() for m in result.messages], [a.to_dict() for a in result.artifacts],
                 [p.to_dict() for p in result.parts])
        if before_copy != after:
            raise AssertionError(f"raw records mutated by feature generation for {outcome.task_instance_id}")

        per_task[outcome.task_instance_id] = features
        all_token_rows.extend(features["token_features"])
        all_timing_rows.extend(features["timing_features"])

    determinism_hash = hashlib.sha256(_canonical_json(per_task).encode("utf-8")).hexdigest()

    null_rate_by_field = {
        "token_features.input_token_count": _null_rate(all_token_rows, "input_token_count"),
        "token_features.output_token_count": _null_rate(all_token_rows, "output_token_count"),
        "token_features.expansion_ratio": _null_rate(all_token_rows, "expansion_ratio"),
        "token_features.ctx_delta": _null_rate(all_token_rows, "ctx_delta"),
        "timing_features.generation_time_ms": _null_rate(all_timing_rows, "generation_time_ms"),
        "timing_features.tokens_per_second": _null_rate(all_timing_rows, "tokens_per_second"),
        "timing_features.generation_ratio": _null_rate(all_timing_rows, "generation_ratio"),
        "timing_features.wall_clock_latency_ms": _null_rate(all_timing_rows, "wall_clock_latency_ms"),
    }

    return {
        "experiment_version": "travel_a2a_v2",
        "step": "Step 7B",
        "session_count": len(outcome_pairs),
        "determinism_hash": determinism_hash,
        "null_rate_by_field": null_rate_by_field,
        "note": (
            "token_features and timing_features' Ollama-duration/token-derived fields "
            "are ~100% null here by design: all 50 sessions ran under "
            "timing_source == 'deterministic_mock' (mock_runner.py never sets "
            "llm_called=True or populates prompt_eval_count/eval_count/eval_duration/"
            "prompt_eval_duration/total_duration) -- this is expected, not a generator "
            "bug. wall_clock_latency_ms and every graph_features/session_features field "
            "ARE fully populated under mock, since they derive from event/message/"
            "artifact structure rather than Ollama telemetry. Real token/timing values "
            "require an ollama_runtime run of this workload -- still "
            "PLANNED_NOT_EXECUTED (configs/travel_a2a/formal_workload/"
            "formal_collection_plan.json)."
        ),
    }


def write_feature_generation_manifest(report_root: str = DEFAULT_REPORT_ROOT) -> Dict[str, Any]:
    manifest = build_feature_generation_manifest()
    os.makedirs(report_root, exist_ok=True)
    with open(os.path.join(report_root, "feature_generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


if __name__ == "__main__":
    result = write_feature_generation_manifest()
    print(json.dumps({k: v for k, v in result.items() if k != "note"}, indent=2))
