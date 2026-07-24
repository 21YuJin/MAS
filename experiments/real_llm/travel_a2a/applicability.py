"""
[Step 6-2] AttackApplicabilityMatrix -- which (attack_family, task_fixture_id)
combinations are valid to run, and what injection_source_id/entry_agent_id
that combination resolves to. A payload-variant TEMPLATE (the JSON files
under configs/travel_a2a/attacks/, e.g. preference_manipulation_v1.json) only
defines WHAT the payload says and WHICH family/evaluator it belongs to; the
matrix says WHERE it's valid to point that payload for a given task (since
option_ids like "H_OSA_2" are destination-specific, a v1 template can't be
reused verbatim across different destinations without knowing which option in
THAT destination's content to target).

build_attack_config_for_task() merges the two: the variant template supplies
the payload/family/evaluator, the matrix row supplies injection_source_id/
entry_agent_id for this specific task -- producing a fresh AttackConfig with
a task-specific attack_id so runs against different fixtures never collide.
"""
import copy
import dataclasses
import json
import os
from typing import List, Optional

from .attack_models import AttackConfig

DEFAULT_MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "configs", "travel_a2a", "attack_applicability_matrix.json")


@dataclasses.dataclass
class ApplicabilityRow:
    attack_family: str
    task_fixture_id: str
    entry_agent_id: str
    injection_source_id: Optional[str]
    evaluator_id: str
    applicable: bool
    exclusion_reason: Optional[str] = None

    def __post_init__(self):
        if self.applicable and not self.injection_source_id:
            raise ValueError(f"row ({self.attack_family}, {self.task_fixture_id}) is marked applicable "
                              f"but has no injection_source_id")
        if not self.applicable and not self.exclusion_reason:
            raise ValueError(f"row ({self.attack_family}, {self.task_fixture_id}) is marked NOT applicable "
                              f"but has no exclusion_reason")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_applicability_matrix(path: str = DEFAULT_MATRIX_PATH) -> List[ApplicabilityRow]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [ApplicabilityRow(**r) for r in rows]


def get_applicable_tasks(matrix: List[ApplicabilityRow], attack_family: str) -> List[ApplicabilityRow]:
    return [r for r in matrix if r.attack_family == attack_family and r.applicable]


def find_row(matrix: List[ApplicabilityRow], attack_family: str, task_fixture_id: str) -> Optional[ApplicabilityRow]:
    for r in matrix:
        if r.attack_family == attack_family and r.task_fixture_id == task_fixture_id:
            return r
    return None


class InapplicableCombinationError(ValueError):
    pass


def build_attack_config_for_task(template: AttackConfig, matrix: List[ApplicabilityRow],
                                  task_fixture_id: str) -> AttackConfig:
    """Raises InapplicableCombinationError for a combination the matrix
    doesn't allow (either no row at all, or a row explicitly marked
    applicable=false) -- callers must never silently skip this check and run
    an attack against a task it was never validated for."""
    row = find_row(matrix, template.attack_family, task_fixture_id)
    if row is None:
        raise InapplicableCombinationError(
            f"no applicability row for ({template.attack_family!r}, {task_fixture_id!r})")
    if not row.applicable:
        raise InapplicableCombinationError(
            f"({template.attack_family!r}, {task_fixture_id!r}) is explicitly NOT applicable: {row.exclusion_reason}")
    if row.entry_agent_id != template.entry_agent_id:
        raise ValueError(f"matrix row entry_agent_id {row.entry_agent_id!r} != template's {template.entry_agent_id!r}")

    data = template.to_dict()
    data["attack_id"] = f"{template.attack_id}__{task_fixture_id}"
    data["injection_source_id"] = row.injection_source_id
    if template.attack_family == "preference_manipulation":
        data["malicious_target_option_id"] = row.injection_source_id
    data["payload_hash"] = None  # recomputed in __post_init__ -- same payload -> same hash regardless of task
    return AttackConfig.from_dict(data)
