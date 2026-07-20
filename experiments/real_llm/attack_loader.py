"""
Attack template loader -- decouples attack scenario definitions from
execution code, so each attack can be described (and cited in a paper) as a
security scenario rather than an ad hoc string buried in a Python file.

Reads every configs/attacks/*.json file, validates the combined template set
(including cross-checking injection_agent/target_agent/expected_propagation_path
against the topology config from experiments/real_llm/config/), and exposes it
as a flat list of dicts.

Required per-template schema (see configs/attacks/*.json):
    attack_type, template_id, injection_agent, target_agent, intensity,
    expected_propagation_path, prompt_template, length_constraint,
    attack_goal, attacker_capability, injection_channel,
    expected_security_impact
"""
import json
import os

ATTACKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "attacks")
TOPOLOGY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "topology_4agent_v1.json")
REQUIRED_ATTACK_TYPES = {"direct", "slow", "chain", "length_preserving"}
VALID_INTENSITIES = {"low", "medium", "high"}
REQUIRED_FIELDS = [
    "attack_type", "template_id", "injection_agent", "target_agent", "intensity",
    "expected_propagation_path", "prompt_template", "length_constraint",
    "attack_goal", "attacker_capability", "injection_channel", "expected_security_impact",
]


def _load_topology_nodes_and_edges(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    nodes = set(cfg["nodes"])
    edges = {frozenset(e) for e in cfg["edges"]}
    return nodes, edges


def load_all_attacks(attacks_dir=ATTACKS_DIR, topology_path=TOPOLOGY_CONFIG_PATH):
    """
    Loads and validates every *.json file in attacks_dir (one JSON list of
    attack-template dicts per file, by convention one file per attack_type).

    Validates:
      - every template has all REQUIRED_FIELDS
      - template_id is globally unique across all files
      - template["attack_type"] matches the file's declared type (filename
        stem), catching copy-paste mistakes across attack-type files
      - intensity is one of {low, medium, high}
      - injection_agent and target_agent are real nodes in the topology config
        (§experiments/real_llm/config/topology_4agent_v1.json)
      - expected_propagation_path starts at injection_agent and ends at
        target_agent, and every consecutive pair in the path is connected by
        an actual edge in the topology -- an attack can't claim to propagate
        along a path that doesn't exist in the graph
      - all 4 required attack types (direct, slow, chain, length_preserving)
        are present

    Returns a flat list of attack-template dicts.
    """
    assert os.path.isdir(attacks_dir), f"attacks directory not found: {attacks_dir}"
    nodes, edges = _load_topology_nodes_and_edges(topology_path)

    all_templates = []
    seen_ids = {}
    for fname in sorted(os.listdir(attacks_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(attacks_dir, fname)
        with open(path, encoding="utf-8") as f:
            templates = json.load(f)
        assert isinstance(templates, list), f"{fname}: expected a JSON list of attack-template objects"

        file_attack_type = os.path.splitext(fname)[0]
        for t in templates:
            missing = [k for k in REQUIRED_FIELDS if k not in t]
            assert not missing, f"{fname}/{t.get('template_id', '?')}: missing field(s) {missing}"

            assert t["attack_type"] == file_attack_type, \
                f"{fname}: template {t['template_id']} has attack_type={t['attack_type']!r}, " \
                f"expected {file_attack_type!r} (file/attack_type mismatch)"

            assert t["template_id"] not in seen_ids, \
                f"duplicate template_id {t['template_id']!r} in {fname} (already seen in {seen_ids[t['template_id']]})"
            seen_ids[t["template_id"]] = fname

            assert t["intensity"] in VALID_INTENSITIES, \
                f"{t['template_id']}: intensity={t['intensity']!r} not in {VALID_INTENSITIES}"

            assert t["injection_agent"] in nodes, \
                f"{t['template_id']}: injection_agent={t['injection_agent']!r} is not a topology node"
            assert t["target_agent"] in nodes, \
                f"{t['template_id']}: target_agent={t['target_agent']!r} is not a topology node"

            path_nodes = t["expected_propagation_path"]
            assert path_nodes, f"{t['template_id']}: expected_propagation_path must be non-empty"
            assert all(n in nodes for n in path_nodes), \
                f"{t['template_id']}: expected_propagation_path contains unknown node(s): {path_nodes}"
            assert path_nodes[0] == t["injection_agent"], \
                f"{t['template_id']}: expected_propagation_path must start at injection_agent " \
                f"({t['injection_agent']!r}), starts at {path_nodes[0]!r}"
            assert path_nodes[-1] == t["target_agent"], \
                f"{t['template_id']}: expected_propagation_path must end at target_agent " \
                f"({t['target_agent']!r}), ends at {path_nodes[-1]!r}"
            for a, b in zip(path_nodes, path_nodes[1:]):
                assert frozenset((a, b)) in edges, \
                    f"{t['template_id']}: expected_propagation_path step {a!r}->{b!r} has no " \
                    f"corresponding edge in the topology"

            all_templates.append(t)

    present_types = {t["attack_type"] for t in all_templates}
    missing_types = REQUIRED_ATTACK_TYPES - present_types
    assert not missing_types, f"missing required attack_type(s): {missing_types}"

    return all_templates


def attack_type_counts(templates):
    counts = {}
    for t in templates:
        counts[t["attack_type"]] = counts.get(t["attack_type"], 0) + 1
    return dict(sorted(counts.items()))


def print_attack_type_counts(templates):
    counts = attack_type_counts(templates)
    print(f"  Loaded {len(templates)} attack templates across {len(counts)} attack types:")
    for atype, n in counts.items():
        print(f"    {atype:<20} {n:3d}")


if __name__ == "__main__":
    templates = load_all_attacks()
    print_attack_type_counts(templates)
    print(f"\n  Required attack types present: {REQUIRED_ATTACK_TYPES}")
    print(f"  Sample template: {templates[0]['template_id']} "
          f"({templates[0]['injection_agent']} -> {templates[0]['target_agent']}, "
          f"intensity={templates[0]['intensity']})")
