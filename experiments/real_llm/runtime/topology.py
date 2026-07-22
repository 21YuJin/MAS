"""
[Step 1-3, shared runtime foundation] Single source for the topology's ordered
agent-name list. lgnn_experiment.py and collect_normal.py already load and
validate the full topology config themselves (adjacency/predecessor checks
included, see their own load_topology()) -- this module is for consumers that
only ever needed the plain name list (localization_analysis.py,
mini_validation.py), which previously hardcoded
["Agent_0", "Agent_1", "Agent_2", "Agent_3"] as a literal instead of reading
config/topology_4agent_v1.json, so a topology change could silently drift out
of sync between files.

Deliberately NOT consolidating lgnn_experiment.py/collect_normal.py's own
(already-correct, already-validating) load_topology() into this module yet --
this is a narrower fix for the two files that had no topology awareness at
all, not a full topology-loader unification.
"""
import json
import os

DEFAULT_TOPOLOGY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "topology_4agent_v1.json")


def load_agent_names(path=DEFAULT_TOPOLOGY_PATH):
    """Ordered node-name list from a topology config -- no adjacency/
    predecessor validation (that's load_topology() in lgnn_experiment.py/
    collect_normal.py, for consumers that actually build a graph from it)."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["nodes"]
