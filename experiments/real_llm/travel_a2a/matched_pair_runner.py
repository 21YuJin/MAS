"""
[Step 5-10/5-12] MatchedPairRunner -- runs a normal session and a matched
attack session against the SAME task fixture, independently verifies
request/base-content equivalence (never trusting that the agents "did the
right thing" -- this check is computed fresh, without depending on anything
the actual session run touched), evaluates the pair via attack_evaluators.py,
and saves the whole bundle.

Deterministic execution order (normal then attack) is used here -- Step 5-10
notes that later work should randomize order to rule out timing bias, but
this smoke test doesn't need that yet.
"""
import dataclasses
import hashlib
import json
import os
from typing import Optional

from .attack_evaluators import ArtifactEvaluator, PairwiseOutcomeEvaluator, StructuralEvaluator, evaluate_attack
from .attack_models import AttackConfig
from .content_repository import ContentRepository
from .fixtures import build_travel_task
from .ids import DeterministicIdFactory
from .injection_builder import build_external_content
from .ollama_runner import run_ollama_workflow
from .session_store import save_session

DEFAULT_ATTACK_SMOKE_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "travel_a2a", "attack_smoke"))

CONTENT_LOOKUP = {
    "flight_agent": lambda repo, dest: repo.flights_for(dest),
    "hotel_agent": lambda repo, dest: repo.hotels_for(dest),
    "tours_agent": lambda repo, dest: repo.tours_for(dest),
}


def _hash_request(request) -> str:
    return hashlib.sha256(json.dumps(request.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass
class MatchedPairResult:
    pair_id: str
    task_fixture_id: str
    normal_session_id: str
    attack_session_id: str
    request_hash_equal: bool
    base_content_hash_equal: bool
    injected_source_id: str
    normal_diagnostics: dict
    attack_diagnostics: dict
    pairwise_differences: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class MatchedPairRunner:
    def __init__(self, content_repository: ContentRepository):
        self.content_repository = content_repository

    def run(self, fixture_dict: dict, attack_config: AttackConfig):
        """Returns (MatchedPairResult, normal_result, attack_result) -- the
        latter two are MockWorkflowResult instances (from ollama_runner.py),
        needed by the caller to actually save full session data (Step 5-12)."""
        fixture_id = fixture_dict["task_fixture_id"]
        # attack_family (not the full attack_id, which already repeats the
        # family name) keeps this short -- see save_matched_pair()'s
        # MAX_PATH note for why path length matters here.
        pair_id = f"pair_{fixture_id}_{attack_config.attack_family}"

        normal_task = build_travel_task(fixture_dict, task_id=f"task_{fixture_id}_normal",
                                         context_id=f"ctx_{fixture_id}_normal")
        attack_task = build_travel_task(fixture_dict, task_id=f"task_{fixture_id}_attack",
                                         context_id=f"ctx_{fixture_id}_attack")
        attack_task.condition = "attack"
        attack_task.injection_present = True
        attack_task.attack_id = attack_config.attack_id

        # [Step 5-10 step 3] Independently verified -- computed fresh from
        # the fixture/content_repository, not read back from anything the
        # actual session run produced.
        request_hash_equal = _hash_request(normal_task.request) == _hash_request(attack_task.request)

        lookup_fn = CONTENT_LOOKUP[attack_config.entry_agent_id]
        records = lookup_fn(self.content_repository, normal_task.request.destination)
        base_record = next(r for r in records if r.get("option_id") == attack_config.injection_source_id)
        normal_ext = build_external_content(base_record, "normal")
        attack_ext = build_external_content(base_record, "attack", attack_config=attack_config)
        base_content_hash_equal = normal_ext.base_content_hash == attack_ext.base_content_hash

        normal_session_id = f"{pair_id}_normal"
        attack_session_id = f"{pair_id}_attack"
        normal_result = run_ollama_workflow(normal_task, self.content_repository,
                                             id_factory=DeterministicIdFactory(), session_id=normal_session_id)
        attack_result = run_ollama_workflow(attack_task, self.content_repository,
                                             id_factory=DeterministicIdFactory(), session_id=attack_session_id,
                                             attack_config=attack_config)

        diagnostics = evaluate_attack(attack_config, normal_result, attack_result, session_id=attack_session_id)

        s_normal = StructuralEvaluator().evaluate(normal_result)
        s_attack = StructuralEvaluator().evaluate(attack_result)
        a_normal = ArtifactEvaluator().evaluate(normal_result)
        a_attack = ArtifactEvaluator().evaluate(attack_result)
        pairwise = PairwiseOutcomeEvaluator().evaluate(s_normal, s_attack, a_normal, a_attack)

        pair_result = MatchedPairResult(
            pair_id=pair_id, task_fixture_id=fixture_id,
            normal_session_id=normal_session_id, attack_session_id=attack_session_id,
            request_hash_equal=request_hash_equal, base_content_hash_equal=base_content_hash_equal,
            injected_source_id=attack_config.injection_source_id,
            normal_diagnostics={"selected_options": a_normal["selected_option_id"], "status": s_normal["final_status"]},
            attack_diagnostics=diagnostics.to_dict(),
            pairwise_differences=pairwise,
        )
        return pair_result, normal_result, attack_result


def save_matched_pair(pair_result: MatchedPairResult, normal_result, attack_result,
                       attack_config: AttackConfig, output_root: str = DEFAULT_ATTACK_SMOKE_ROOT) -> str:
    """[Step 5-12] outputs/travel_a2a/attack_smoke/<pair_id>/{normal,attack}/
    (full session bundles via session_store.save_session) plus the
    pair-level summary files at the top of the pair directory. pairwise_diff
    stores DIFFS ONLY (selected-option/status/event-pattern deltas), never
    raw response text -- raw evidence lives exclusively inside normal/ and
    attack/'s own raw (non-metadata) files."""
    pair_dir = os.path.join(output_root, pair_result.pair_id)
    os.makedirs(pair_dir, exist_ok=True)

    # save_session()'s `session_id` argument is only used for directory
    # naming (session_dir_for()) -- it is NOT re-injected into any record
    # (each record already carries its own real session_id from the actual
    # run). Passing the short literal "normal"/"attack" here keeps the path
    # well under Windows' MAX_PATH, instead of nesting the full (long)
    # pair_id-based session_id as its own directory level.
    save_session("normal", normal_result.task, normal_result.messages, normal_result.parts,
                 normal_result.artifacts, normal_result.events, agent_call_records=normal_result.agent_call_records,
                 output_root=pair_dir)
    save_session("attack", attack_result.task, attack_result.messages, attack_result.parts,
                 attack_result.artifacts, attack_result.events, agent_call_records=attack_result.agent_call_records,
                 output_root=pair_dir)

    with open(os.path.join(pair_dir, "attack_config.json"), "w", encoding="utf-8") as f:
        json.dump(attack_config.to_dict(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(pair_dir, "matched_pair_result.json"), "w", encoding="utf-8") as f:
        json.dump(pair_result.to_dict(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(pair_dir, "attack_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(pair_result.attack_diagnostics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(pair_dir, "evaluator_report.json"), "w", encoding="utf-8") as f:
        json.dump(pair_result.attack_diagnostics.get("evaluator_evidence", {}), f, indent=2, ensure_ascii=False)
    with open(os.path.join(pair_dir, "pairwise_diff.json"), "w", encoding="utf-8") as f:
        json.dump(pair_result.pairwise_differences, f, indent=2, ensure_ascii=False)

    return pair_dir
