"""
Generates matched normal/attack session SPECS from the same base task, so a
downstream session generator can produce directly-comparable pairs instead of
independently-sampled normal and attack pools.

This is a spec generator only -- it does not call Ollama. It produces the
manifest a future session generator would consume (see README §공격 시나리오
객관화); wiring it into lgnn_experiment.py's actual collection loop is the
next step (real data recollection), not this one.

Each pair shares a pair_id ("{base_task_id}_seed_{seed}"); the two members are
flat records distinguished by injection_enabled, matching the minimal schema
from the original spec:
    {"pair_id": ..., "base_task_id": ..., "injection_enabled": bool, ...}

The attack member also carries attack_type/attack_template_id/intensity_intended
so that once a session actually runs, the eventual session-metadata record can
store BOTH the intended intensity (from this spec) and the actually-observed
metrics (attack_success_observed, token counts, etc.) side by side -- neither
overwrites the other.
"""
import itertools
import json
import os

from task_loader import load_all_tasks, tasks_by_id
from attack_loader import load_all_attacks


def generate_pairs(task_ids, seeds, templates, assignment="round_robin"):
    """
    For every (task_id, seed) combination, returns two flat session-spec
    records sharing one pair_id: one normal (injection_enabled=False), one
    attack (injection_enabled=True, using a template chosen from `templates`).

    assignment="round_robin": cycles through `templates` in a fixed, seed-
    independent order so which template a given pair gets is deterministic
    and reproducible from (task_ids, seeds, templates) alone.
    """
    assert templates, "need at least one attack template to pair against"
    specs = []
    template_cycle = itertools.cycle(templates)
    for task_id in task_ids:
        for seed in seeds:
            pair_id = f"{task_id}_seed_{seed}"
            specs.append({
                "pair_id": pair_id,
                "base_task_id": task_id,
                "generation_seed": seed,
                "injection_enabled": False,
                "attack_type": None,
                "attack_template_id": None,
                "intensity_intended": None,
            })
            template = next(template_cycle)
            specs.append({
                "pair_id": pair_id,
                "base_task_id": task_id,
                "generation_seed": seed,
                "injection_enabled": True,
                "attack_type": template["attack_type"],
                "attack_template_id": template["template_id"],
                "intensity_intended": template["intensity"],
            })
    return specs


def validate_pairs(specs):
    """Every pair_id must have exactly one normal member and one attack
    member sharing the same base_task_id and generation_seed."""
    by_pair = {}
    for s in specs:
        by_pair.setdefault(s["pair_id"], []).append(s)
    for pair_id, members in by_pair.items():
        assert len(members) == 2, f"{pair_id}: expected 2 members, got {len(members)}"
        flags = sorted(m["injection_enabled"] for m in members)
        assert flags == [False, True], f"{pair_id}: expected one normal + one attack member, got {flags}"
        base_ids = {m["base_task_id"] for m in members}
        seeds    = {m["generation_seed"] for m in members}
        assert len(base_ids) == 1, f"{pair_id}: members disagree on base_task_id: {base_ids}"
        assert len(seeds) == 1, f"{pair_id}: members disagree on generation_seed: {seeds}"
    return by_pair


if __name__ == "__main__":
    tasks = load_all_tasks()
    templates = load_all_attacks()
    by_id = tasks_by_id(tasks)

    # Dry-run demonstration only: pairs 3 sample task_ids (one per split isn't
    # meaningful here since this doesn't read normal_task_split_v1.json -- the
    # actual task_id list used for real collection is a step-12 decision) x 2
    # seeds against all loaded attack templates, then validates the pairing
    # invariants below. Not written to data/ -- this is a self-test, not the
    # final collection manifest.
    sample_task_ids = sorted(by_id.keys())[:3]
    sample_seeds = [42, 43]

    specs = generate_pairs(sample_task_ids, sample_seeds, templates)
    by_pair = validate_pairs(specs)

    print(f"  Generated {len(specs)} session specs ({len(by_pair)} pairs) "
          f"from {len(sample_task_ids)} tasks x {len(sample_seeds)} seeds")
    for pair_id, members in list(by_pair.items())[:2]:
        normal = next(m for m in members if not m["injection_enabled"])
        attack = next(m for m in members if m["injection_enabled"])
        print(f"    {pair_id}:")
        print(f"      normal: {normal}")
        print(f"      attack: {attack}")
    print("\n  All pairs validated: exactly one normal + one attack member per pair_id,"
          " matching base_task_id and generation_seed.")
