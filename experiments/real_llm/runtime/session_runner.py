"""
[Step 1-4, shared runtime foundation] Common execution-result type + runner
interface that a future workflow-policy-driven execution engine (the planned
travel_a2a_v2 A2A-inspired framework) will produce -- and that today's legacy
4-agent run_session() implementations (lgnn_experiment.py, collect_normal.py)
can already be adapted to fill, without replacing run_session() itself yet.

Scope boundary (per the Step 1 instruction): this module defines the
interface/type boundary only.
  - No Task/Message/Part/Artifact model is implemented here.
  - No InteractionEvent model is implemented here.
  - No new execution engine is implemented here.
  - The existing hardcoded 4-step prompt chain in run_session() is NOT
    touched or replaced by this file.
Those all come later. messages/artifacts/interaction_events on
SessionRunResult are deliberately left as empty lists when produced by
LegacyRunSessionAdapter below, because the legacy pipeline has no Message/
Artifact concept -- it only has raw prompt strings and raw response text.
Populating those three fields for real is exactly what a future execution
engine needs to do; this file does not attempt to fake it.
"""
import abc
import dataclasses
from typing import Any, Callable, Optional


@dataclasses.dataclass
class SessionRunResult:
    """
    Common execution-result envelope. Every field is meant to eventually be
    populated by a real (travel_a2a_v2) execution engine; a legacy-run_session
    adapter can only fill a subset today -- unfillable fields stay at their
    default (empty list / None), never fabricated.
    """
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    # [travel_a2a_v2] groups related sessions/turns under one task/context.
    # The legacy pipeline has no such concept -- always None from the adapter.
    context_id: Optional[str] = None
    # One raw telemetry record per LLM call, same shape as run_session()'s
    # session_telemetry list (analysis_plan.md §4 schema).
    agent_call_records: list = dataclasses.field(default_factory=list)
    # [travel_a2a_v2] Message/Part objects -- not yet modeled. Always [] here.
    messages: list = dataclasses.field(default_factory=list)
    # [travel_a2a_v2] Artifact objects -- not yet modeled. Always [] here.
    artifacts: list = dataclasses.field(default_factory=list)
    # [travel_a2a_v2] InteractionEvent log -- not yet modeled. Always [] here.
    interaction_events: list = dataclasses.field(default_factory=list)
    final_output: Optional[str] = None
    # Whatever diagnostic values the caller already computes today
    # (indicator_observed / goal_success / propagation_observed, etc.) --
    # a free-form dict rather than fixed fields, since which diagnostics
    # exist is still evolving (analysis_plan.md §3).
    diagnostic_labels: dict = dataclasses.field(default_factory=dict)
    errors: list = dataclasses.field(default_factory=list)


class SessionRunner(abc.ABC):
    """
    Interface a future execution engine implements. `condition` is "normal" or
    "attack" (matches the existing convention throughout this codebase);
    `attack_config` is one of configs/attacks/v2/*.json's dicts, or None for
    a normal session.
    """

    @abc.abstractmethod
    def run(self, task, condition, attack_config=None, **kwargs) -> SessionRunResult:
        raise NotImplementedError


class LegacyRunSessionAdapter(SessionRunner):
    """
    Wraps an existing run_session()-shaped function (lgnn_experiment.py's
    run_session, or any function with the same
    (task, injection=..., ...) -> (X, indicator_observed, session_ok,
    session_telemetry) signature) behind the SessionRunner interface, so code
    written against SessionRunner/SessionRunResult already works today
    without waiting for the travel_a2a_v2 execution engine.

    Does not change run_session()'s behavior or output -- purely a wrapping
    layer. Not yet wired into lgnn_experiment.py/collect_normal.py's actual
    collection loops (Step 1 defines the boundary; rewiring collection to go
    through it is a later step's decision, not made here).
    """

    def __init__(self, run_session_fn: Callable[..., Any]):
        """run_session_fn is injected rather than imported directly, so this
        adapter doesn't hardcode a dependency on one specific script's
        implementation (lgnn_experiment.run_session vs. a future variant)."""
        self._run_session_fn = run_session_fn

    def run(self, task, condition, attack_config=None, **kwargs) -> SessionRunResult:
        if condition not in ("normal", "attack"):
            raise ValueError(f"condition must be 'normal' or 'attack', got {condition!r}")

        injection = None
        attack_type = kwargs.pop("attack_type", None)
        attack_goal = kwargs.pop("attack_goal", None)
        if condition == "attack":
            if attack_config is None:
                raise ValueError("condition='attack' requires attack_config")
            injection = attack_config.get("injection_template")
            attack_type = attack_type or attack_config.get("attack_id")
            attack_goal = attack_goal or attack_config.get("attack_goal")

        _X, indicator_observed, session_ok, session_telemetry = self._run_session_fn(
            task, injection=injection, attack_type=attack_type, attack_goal=attack_goal, **kwargs)

        final_output = session_telemetry[-1]["response_text"] if session_telemetry else None

        return SessionRunResult(
            session_id=kwargs.get("session_id"),
            task_id=kwargs.get("task_id"),
            context_id=None,
            agent_call_records=session_telemetry,
            messages=[],
            artifacts=[],
            interaction_events=[],
            final_output=final_output,
            diagnostic_labels={"indicator_observed": indicator_observed},
            errors=([] if session_ok else
                    ["one or more agent calls failed or returned an empty response"]),
        )
