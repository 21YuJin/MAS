"""
[Step 6.5D] Formal workload mock full-run and behavioral validation -- runs
all 50 formal TaskInstances (normal condition, 1 repeat each, deterministic
mock -- no Ollama) through the SAME run_mock_workflow()/TravelWorkflowPolicy
used for the 6 development fixtures, then checks whether the workload's
STATIC diversity (Step 6.5B/C) actually produces diverse EXECUTION behavior:
branch match against expected_normal_branches, event/graph pattern
diversity, difficulty/split behavioral confounds, active-agent coverage, and
LightGAE-input readiness (active_agent_mask / adjacency signature only --
no feature vectors, no scaler, no training).
"""
import dataclasses
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .agents import MODEL_AGENT_ORDER, build_default_registry
from .content_repository import ContentRepository, load_content_repository
from .formal_workload_generator import OUTPUT_DIR_DEFAULT
from .formal_workload_models import TaskInstance
from .formal_workload_validation import load_task_instances
from .mock_runner import DeterministicClock, run_mock_workflow
from .models import ArtifactType, FORBIDDEN_METADATA_KEYS, InteractionType, TravelRequest, TravelTask
from .session_store import save_session
from .status import TaskStatus
from .validation import (
    validate_artifact, validate_artifact_lineage, validate_context_consistency, validate_event,
    validate_event_sequence, validate_message,
)

FORMAL_CONTENT_DIR = os.path.join(OUTPUT_DIR_DEFAULT, "content")
MOCK_OUTPUT_ROOT_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "travel_a2a", "formal_workload_mock"))
REPORT_ROOT_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reports", "travel_a2a", "formal_workload"))

FIXTURE_CREATED_AT = "2027-01-01T00:00:00+00:00"

# task_group_id/split/hard_normal_tags/generation_seed/generator_version are
# diagnostic-only (Step 6.5-2) -- never placed in TravelTask.provenance's
# LLM-facing surface. provenance IS diagnostic-only by construction already
# (fixtures.py does the same for task_fixture_id), so it's the right place.


def build_formal_travel_task(ti: TaskInstance, task_id: str, context_id: str,
                              created_at: str = FIXTURE_CREATED_AT) -> TravelTask:
    request = TravelRequest(**ti.to_travel_request_kwargs())
    return TravelTask(
        task_id=task_id, context_id=context_id, request=request,
        status=TaskStatus.SUBMITTED, condition="normal", injection_present=False, attack_id=None,
        created_at=created_at, updated_at=created_at,
        provenance={"task_instance_id": ti.task_instance_id, "task_group_id": ti.task_group_id,
                    "template_id": ti.template_id},
    )


def load_formal_content_repository(content_dir: str = FORMAL_CONTENT_DIR) -> ContentRepository:
    return load_content_repository(base_dir=content_dir)


# ══════════════════════════════════════════════════════════════════════════
# Branch observation -- derived from the SAME triggers workflow_policy.py
# actually uses, not re-guessed independently.
# ══════════════════════════════════════════════════════════════════════════


def _observed_branches(messages) -> List[str]:
    observed = []
    if any(m.interaction_type == InteractionType.REVISION_REQUEST and m.receiver_id == "hotel_agent" for m in messages):
        observed.append("budget_revision")
    if any(m.interaction_type == InteractionType.CLARIFICATION_REQUEST and m.sender_id == "tours_agent"
           and m.receiver_id == "flight_agent" for m in messages):
        observed.append("schedule_clarification")
    if any(m.interaction_type == InteractionType.REVISION_REQUEST and m.receiver_id == "tours_agent" for m in messages):
        observed.append("integration_revision")
    if any(m.interaction_type == InteractionType.CLARIFICATION_REQUEST and m.sender_id == "travel_coordinator"
           and m.receiver_id == "client" for m in messages):
        observed.append("client_clarification")
    if not observed:
        observed.append("basic_flow_only")
    if len(observed) >= 2:
        observed.append("multi_branch")
    return observed


# ══════════════════════════════════════════════════════════════════════════
# Signatures -- event pattern / graph pattern / adjacency (LightGAE-input readiness)
# ══════════════════════════════════════════════════════════════════════════


def event_pattern_signature(events) -> Tuple[Tuple[str, str, str], ...]:
    return tuple((e.sender_id, e.receiver_id, e.interaction_type.value) for e in events)


def graph_pattern_signature(events) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, int], ...]]:
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for e in events:
        pair_counts[(e.sender_id, e.receiver_id)] += 1
    unique_pairs = tuple(sorted(f"{a}->{b}" for a, b in pair_counts))
    repeated_pair_counts = tuple(sorted((f"{a}->{b}", n) for (a, b), n in pair_counts.items() if n > 1))
    return unique_pairs, repeated_pair_counts


def active_agent_mask(messages) -> Dict[str, bool]:
    active = {m.sender_id for m in messages} | {m.receiver_id for m in messages}
    return {agent_id: (agent_id in active) for agent_id in MODEL_AGENT_ORDER}


def adjacency_signature(events) -> List[List[str]]:
    """[Step 6.5D LightGAE-readiness check] unique directed (sender,
    receiver) pairs restricted to MODEL_AGENT_ORDER agents -- confirms an
    event-derived adjacency CAN be built and differs from a fixed topology,
    without constructing any tensor here."""
    pairs = sorted({(e.sender_id, e.receiver_id) for e in events
                    if e.sender_id in MODEL_AGENT_ORDER and e.receiver_id in MODEL_AGENT_ORDER})
    return [list(p) for p in pairs]


# ══════════════════════════════════════════════════════════════════════════
# Per-task execution + validation
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class TaskMockRunOutcome:
    task_instance_id: str
    task_group_id: str
    split: Optional[str]
    template_family: str
    difficulty: str
    expected_normal_branches: List[str]
    observed_branches: List[str]
    branch_exact_match: bool
    missing_expected_branches: List[str]
    unexpected_branches: List[str]
    active_agents: List[str]
    event_count: int
    unique_directed_pair_count: int
    repeated_pair_count: int
    message_count: int
    artifact_count: int
    artifact_version_count: int
    clarification_count: int
    revision_count: int
    task_status: str
    final_travel_plan_present: bool
    strict_validation_error_count: int
    strict_validation_errors: List[str]
    event_pattern: Tuple[Tuple[str, str, str], ...]
    active_agent_mask: Dict[str, bool]
    adjacency_signature: List[List[str]]

    def to_summary_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("event_pattern")  # not CSV/JSON-summary friendly at top level -- kept in event_pattern_report.json
        d.pop("strict_validation_errors")
        return d


def run_one_formal_task(ti: TaskInstance, content_repo: ContentRepository, split_by_id: Dict[str, str],
                         registry) -> Tuple[TaskMockRunOutcome, Any]:
    from .ids import DeterministicIdFactory
    task = build_formal_travel_task(ti, task_id=f"task_{ti.task_instance_id}", context_id=f"ctx_{ti.task_instance_id}")
    result = run_mock_workflow(task, content_repo, id_factory=DeterministicIdFactory(),
                                clock=DeterministicClock(), session_id=ti.task_instance_id)

    strict_errors: List[str] = []
    try:
        for m in result.messages:
            validate_message(m, registry, mode="strict")
        for a in result.artifacts:
            validate_artifact(a, registry, mode="strict")
        for e in result.events:
            validate_event(e, registry, mode="strict")
        validate_context_consistency(task, result.messages, result.artifacts, result.events,
                                      parts=result.parts, mode="strict")
        validate_artifact_lineage(result.artifacts, mode="strict")
        validate_event_sequence(result.events, mode="strict")
    except Exception as e:  # noqa: BLE001 -- strict validators raise ValidationError; capture, never abort the run
        strict_errors.append(f"{type(e).__name__}: {e}")

    observed = _observed_branches(result.messages)
    expected = list(ti.expected_normal_branches)
    missing = [b for b in expected if b not in observed]
    unexpected = [b for b in observed if b not in expected]

    final_plan = any(a.artifact_type == ArtifactType.FINAL_TRAVEL_PLAN for a in result.artifacts)
    unique_pairs, repeated_pairs = graph_pattern_signature(result.events)
    clarification_count = sum(1 for m in result.messages if m.interaction_type in
                               (InteractionType.CLARIFICATION_REQUEST, InteractionType.CLARIFICATION_RESPONSE))
    revision_count = sum(1 for m in result.messages if m.interaction_type == InteractionType.REVISION_REQUEST)

    outcome = TaskMockRunOutcome(
        task_instance_id=ti.task_instance_id, task_group_id=ti.task_group_id,
        split=split_by_id.get(ti.task_instance_id), template_family=ti.task_category, difficulty=ti.difficulty,
        expected_normal_branches=expected, observed_branches=observed,
        branch_exact_match=(set(expected) == set(observed)),
        missing_expected_branches=missing, unexpected_branches=unexpected,
        active_agents=sorted({m.sender_id for m in result.messages} | {m.receiver_id for m in result.messages}),
        event_count=len(result.events), unique_directed_pair_count=len(unique_pairs),
        repeated_pair_count=len(repeated_pairs), message_count=len(result.messages),
        artifact_count=len(result.artifacts), artifact_version_count=sum(a.version for a in result.artifacts),
        clarification_count=clarification_count, revision_count=revision_count,
        task_status=task.status.value, final_travel_plan_present=final_plan,
        strict_validation_error_count=len(strict_errors), strict_validation_errors=strict_errors,
        event_pattern=event_pattern_signature(result.events),
        active_agent_mask=active_agent_mask(result.messages), adjacency_signature=adjacency_signature(result.events),
    )
    return outcome, result


def run_all_formal_mock_sessions(output_root: str = MOCK_OUTPUT_ROOT_DEFAULT,
                                  save_sessions: bool = True) -> List[Tuple[TaskMockRunOutcome, Any]]:
    task_instances = load_task_instances()
    content_repo = load_formal_content_repository()
    registry = build_default_registry()

    from .formal_workload_validation import SPLITS_DIR_DEFAULT
    with open(os.path.join(SPLITS_DIR_DEFAULT, "primary_group_split.json"), encoding="utf-8") as f:
        primary_split = json.load(f)
    split_by_id = {}
    for s in ("train", "validation", "test"):
        for tid in primary_split[f"{s}_task_ids"]:
            split_by_id[tid] = s

    outcomes = []
    for ti in task_instances:
        outcome, result = run_one_formal_task(ti, content_repo, split_by_id, registry)
        if save_sessions:
            save_session(ti.task_instance_id, result.task, result.messages, result.parts, result.artifacts,
                         result.events, agent_call_records=result.agent_call_records, output_root=output_root)
        outcomes.append((outcome, result))
    return outcomes


# ══════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ("event_count", "message_count", "artifact_count", "artifact_version_count",
                "clarification_count", "revision_count", "unique_directed_pair_count")


def aggregate_mock_execution_summary(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    completed = sum(1 for o in outcomes if o.task_status == "completed")
    strict_errors = sum(1 for o in outcomes if o.strict_validation_error_count > 0)
    orphan_or_lineage_errors = sum(
        1 for o in outcomes if any("Lineage" in e or "Context" in e or "orphan" in e.lower()
                                    for e in o.strict_validation_errors))
    final_plan_missing = sum(1 for o in outcomes if not o.final_travel_plan_present)
    return {
        "session_count": len(outcomes), "completed_count": completed,
        "strict_validation_error_count": strict_errors,
        "orphan_or_lineage_error_count": orphan_or_lineage_errors,
        "final_travel_plan_missing_count": final_plan_missing,
        "completion_rate": completed / len(outcomes) if outcomes else 0.0,
    }


def aggregate_branch_match(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    mismatches = [o for o in outcomes if not o.branch_exact_match]
    return {
        "exact_match_count": len(outcomes) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": [{"task_instance_id": o.task_instance_id, "expected": o.expected_normal_branches,
                         "observed": o.observed_branches, "missing": o.missing_expected_branches,
                         "unexpected": o.unexpected_branches} for o in mismatches],
    }


def aggregate_event_patterns(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    pattern_counts: Dict[Tuple, int] = defaultdict(int)
    for o in outcomes:
        pattern_counts[o.event_pattern] += 1
    return {
        "distinct_pattern_count": len(pattern_counts),
        "patterns": [{"pattern": [list(step) for step in pattern], "count": n}
                      for pattern, n in sorted(pattern_counts.items(), key=lambda kv: -kv[1])],
        "meets_minimum_10": len(pattern_counts) >= 10,
    }


def aggregate_graph_patterns(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    pattern_counts: Dict[Tuple, int] = defaultdict(int)
    pattern_examples: Dict[Tuple, dict] = {}
    for o in outcomes:
        key = (tuple(sorted(k for k, v in o.active_agent_mask.items() if v)),
               tuple(tuple(p) for p in o.adjacency_signature))
        pattern_counts[key] += 1
        pattern_examples.setdefault(key, {"active_agents": list(key[0]), "adjacency": [list(p) for p in key[1]]})
    return {
        "distinct_pattern_count": len(pattern_counts),
        "patterns": [{**pattern_examples[k], "count": n} for k, n in sorted(pattern_counts.items(), key=lambda kv: -kv[1])],
        "meets_minimum_6": len(pattern_counts) >= 6,
    }


def _metric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 2), "median": round(statistics.median(values), 2),
        "std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def aggregate_difficulty_behavior(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    by_difficulty: Dict[str, List[TaskMockRunOutcome]] = defaultdict(list)
    for o in outcomes:
        by_difficulty[o.difficulty].append(o)

    stats = {d: {m: _metric_stats([getattr(o, m) for o in os_]) for m in _METRIC_KEYS}
              for d, os_ in by_difficulty.items()}

    confounds = []
    order = ["easy", "medium", "hard"]
    present = [d for d in order if d in stats]
    for metric, code in (("event_count", "DIFFICULTY_EVENT_COUNT_CONFOUND"),
                         ("message_count", "DIFFICULTY_MESSAGE_COUNT_CONFOUND"),
                         ("revision_count", "DIFFICULTY_REVISION_CONFOUND")):
        ranges = {d: (stats[d][metric]["min"], stats[d][metric]["max"]) for d in present}
        # non-overlapping ranges across ALL difficulty tiers -> the metric alone would perfectly reveal difficulty
        sorted_present = sorted(present, key=lambda d: order.index(d))
        non_overlapping = True
        for i in range(len(sorted_present) - 1):
            lo_max = ranges[sorted_present[i]][1]
            hi_min = ranges[sorted_present[i + 1]][0]
            if lo_max is None or hi_min is None or lo_max >= hi_min:
                non_overlapping = False
                break
        if non_overlapping and len(sorted_present) >= 2:
            confounds.append({
                "issue_code": code, "metric": metric, "ranges": ranges,
                "explanation": f"{metric} ranges are entirely non-overlapping across difficulty tiers "
                                f"({', '.join(f'{d}={ranges[d]}' for d in sorted_present)}) -- this metric alone "
                                "would let a detector infer task difficulty without reading any injection-related signal.",
            })
    return {"stats_by_difficulty": stats, "confounds": confounds}


def aggregate_split_behavior(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    by_split: Dict[str, List[TaskMockRunOutcome]] = defaultdict(list)
    for o in outcomes:
        if o.split:
            by_split[o.split].append(o)

    stats = {s: {m: _metric_stats([getattr(o, m) for o in os_]) for m in _METRIC_KEYS}
              for s, os_ in by_split.items()}

    warnings = []
    event_means = {s: stats[s]["event_count"]["mean"] for s in stats if stats[s]["event_count"]["mean"] is not None}
    if len(event_means) >= 2:
        all_event_counts = [o.event_count for o in outcomes if o.split]
        overall_std = statistics.pstdev(all_event_counts) if len(all_event_counts) > 1 else 0.0
        max_mean, min_mean = max(event_means.values()), min(event_means.values())
        if overall_std and (max_mean - min_mean) > overall_std:
            warnings.append(f"split mean event_count differs by {max_mean - min_mean:.2f}, "
                             f"more than the overall population std ({overall_std:.2f})")

    branch_rates: Dict[str, Dict[str, float]] = {}
    for s, os_ in by_split.items():
        counts: Dict[str, int] = defaultdict(int)
        for o in os_:
            for b in o.observed_branches:
                counts[b] += 1
        branch_rates[s] = {b: round(n / len(os_), 3) for b, n in counts.items()}
    branches_all = {b for d in branch_rates.values() for b in d}
    for b in branches_all:
        rates = {s: branch_rates[s].get(b, 0.0) for s in branch_rates}
        if rates and (max(rates.values()) - min(rates.values())) > 0.3:
            warnings.append(f"branch '{b}' rate differs by >30 percentage points across splits: {rates}")

    graph_patterns_by_split = {}
    for s, os_ in by_split.items():
        keys = {(tuple(sorted(k for k, v in o.active_agent_mask.items() if v)),
                 tuple(tuple(p) for p in o.adjacency_signature)) for o in os_}
        graph_patterns_by_split[s] = len(keys)
    test_only_patterns = None
    if "test" in by_split and "train" in by_split:
        train_keys = {(tuple(sorted(k for k, v in o.active_agent_mask.items() if v)),
                       tuple(tuple(p) for p in o.adjacency_signature)) for o in by_split["train"]}
        test_keys = {(tuple(sorted(k for k, v in o.active_agent_mask.items() if v)),
                      tuple(tuple(p) for p in o.adjacency_signature)) for o in by_split.get("test", [])}
        unseen_in_test = test_keys - train_keys
        if unseen_in_test:
            warnings.append(f"{len(unseen_in_test)} graph pattern(s) appear in test but never in train "
                             "(reported only -- not auto-failed per Step 6.5D instructions)")

    return {"stats_by_split": stats, "branch_rates_by_split": branch_rates,
            "graph_pattern_count_by_split": graph_patterns_by_split, "warnings": warnings}


_ACTIVE_AGENT_MINIMUMS = {"travel_coordinator": 50, "flight_agent": 40, "hotel_agent": 40,
                           "currency_agent": 30, "tours_agent": 30}


def aggregate_active_agent_report(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    counts = {a: 0 for a in MODEL_AGENT_ORDER}
    for o in outcomes:
        for a in o.active_agents:
            if a in counts:
                counts[a] += 1
    return {
        "active_session_counts": counts, "total_sessions": len(outcomes),
        "minimums": dict(_ACTIVE_AGENT_MINIMUMS),
        "below_minimum": {a: counts[a] for a, m in _ACTIVE_AGENT_MINIMUMS.items() if counts[a] < m},
    }


def aggregate_hard_budget_confound_report(outcomes: List[TaskMockRunOutcome]) -> Dict[str, Any]:
    hard = [o for o in outcomes if o.difficulty == "hard"]
    branch_counts: Dict[str, int] = defaultdict(int)
    for o in hard:
        for b in o.observed_branches:
            branch_counts[b] += 1
    event_patterns = defaultdict(int)
    for o in hard:
        event_patterns[o.event_pattern] += 1
    dominant_pattern_share = (max(event_patterns.values()) / len(hard)) if hard else 0.0
    budget_only_separates = None
    if hard:
        with_budget = {o.task_instance_id for o in hard if "budget_revision" in o.observed_branches}
        budget_only_separates = len(with_budget) / len(hard)
    return {
        "hard_task_count": len(hard),
        "observed_branch_counts": dict(branch_counts),
        "dominant_event_pattern_share": round(dominant_pattern_share, 3),
        "budget_revision_rate_among_hard": round(budget_only_separates, 3) if budget_only_separates is not None else None,
        "recommend_redesign": bool(hard) and (dominant_pattern_share >= 0.70 or
                                                (budget_only_separates is not None and budget_only_separates >= 0.95)),
    }


# ══════════════════════════════════════════════════════════════════════════
# Report writers
# ══════════════════════════════════════════════════════════════════════════

REPORT_ROOT_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reports", "travel_a2a", "formal_workload"))


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: List[dict]) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_phase_6_5d_reports(outcomes: List[TaskMockRunOutcome], report_root: str = REPORT_ROOT_DEFAULT) -> dict:
    exec_summary = aggregate_mock_execution_summary(outcomes)
    branch_match = aggregate_branch_match(outcomes)
    event_patterns = aggregate_event_patterns(outcomes)
    graph_patterns = aggregate_graph_patterns(outcomes)
    difficulty_behavior = aggregate_difficulty_behavior(outcomes)
    split_behavior = aggregate_split_behavior(outcomes)
    active_agent = aggregate_active_agent_report(outcomes)
    hard_budget_confound = aggregate_hard_budget_confound_report(outcomes)

    _write_json(os.path.join(report_root, "mock_execution_summary.json"), exec_summary)
    _write_csv(os.path.join(report_root, "mock_execution_summary.csv"), [o.to_summary_dict() for o in outcomes])
    _write_json(os.path.join(report_root, "branch_match_report.json"), branch_match)
    _write_json(os.path.join(report_root, "event_pattern_report.json"), event_patterns)
    _write_json(os.path.join(report_root, "graph_pattern_report.json"), graph_patterns)
    _write_json(os.path.join(report_root, "difficulty_behavior_report.json"), difficulty_behavior)
    _write_json(os.path.join(report_root, "split_behavior_report.json"), split_behavior)
    _write_json(os.path.join(report_root, "active_agent_report.json"), active_agent)
    _write_json(os.path.join(report_root, "hard_budget_confound_report.json"), hard_budget_confound)

    validation_report = {
        "phase": "6.5D_mock_full_run_and_behavioral_validation",
        "session_count": exec_summary["session_count"],
        "completed_count": exec_summary["completed_count"],
        "strict_validation_error_count": exec_summary["strict_validation_error_count"],
        "branch_mismatch_count": branch_match["mismatch_count"],
        "distinct_event_pattern_count": event_patterns["distinct_pattern_count"],
        "distinct_graph_pattern_count": graph_patterns["distinct_pattern_count"],
        "difficulty_confounds_found": [c["issue_code"] for c in difficulty_behavior["confounds"]],
        "active_agent_below_minimum": active_agent["below_minimum"],
        "split_behavior_warnings": split_behavior["warnings"],
        "hard_budget_confound_recommend_redesign": hard_budget_confound["recommend_redesign"],
        "overall_pass": (
            exec_summary["completed_count"] == exec_summary["session_count"]
            and exec_summary["strict_validation_error_count"] == 0
            and branch_match["mismatch_count"] == 0
            and event_patterns["meets_minimum_10"]
            and graph_patterns["meets_minimum_6"]
        ),
    }
    _write_json(os.path.join(report_root, "phase_6_5d_validation_report.json"), validation_report)
    return validation_report
