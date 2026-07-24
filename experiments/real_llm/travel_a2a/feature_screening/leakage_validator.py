"""
[Phase 7D-E] Leakage validator -- scans every registry entry's
computation-defining text (feature_name, formula, source_fields) for a
models.FORBIDDEN_METADATA_KEYS term or a dataset-provenance term. Never
mutates the registry -- report only.

Deliberately excluded from the scan: known_confound, leakage_risk, notes.
known_confound/leakage_risk are annotation fields whose whole purpose is to
NAME a confound risk like "difficulty" -- not an accidental leak of it as a
computation input. notes is free-text commentary that may legitimately
discuss split/attack-data POLICY (e.g. "must be fit on the train split's
normal sessions only") without that policy statement being a source field --
per the Phase 7D spec, only formula/source_fields must never reference these
terms as actual computation input.
"""
import re
from typing import Any, Dict, List

from ..models import FORBIDDEN_METADATA_KEYS

_PROVENANCE_TERMS = {
    "difficulty", "split", "expected_normal_branches", "task_group_id", "generation_seed", "hard_normal_tags",
}
FORBIDDEN_TERMS = FORBIDDEN_METADATA_KEYS | _PROVENANCE_TERMS

_TEXT_FIELDS = ("feature_name", "formula")


def _entry_text(entry: Dict[str, Any]) -> str:
    parts = [str(entry.get(field, "")) for field in _TEXT_FIELDS]
    parts.extend(str(s) for s in entry.get("source_fields", []))
    return " ".join(parts).lower()


def scan_feature_for_leakage(entry: Dict[str, Any]) -> List[str]:
    text = _entry_text(entry)
    return sorted(term for term in FORBIDDEN_TERMS if re.search(rf"\b{re.escape(term)}\b", text))


def validate_no_leakage(registry: Dict[str, Any]) -> Dict[str, Any]:
    findings = []
    for entry in registry["features"]:
        hits = scan_feature_for_leakage(entry)
        if hits:
            findings.append({"feature_name": entry["feature_name"], "forbidden_terms_found": hits})
    return {"findings": findings, "passed": not findings}
