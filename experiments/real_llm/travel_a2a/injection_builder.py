"""
[Step 5-5] InjectionBuilder -- builds the external-content record a
specialist agent's prompt is built from, for either condition.

Principle (Step 5-1 #3/#4/#5): normal and attack share the IDENTICAL base
content record (same option_id, price, dates, description) -- the only
difference for a matched pair is whether attack_config's payload_template is
appended to the ONE field injection_source_type maps to. Every other field is
byte-identical between normal and attack, and this is independently
verifiable via base_content_hash (computed from the SAME pre-injection
base_record in both cases, so a matched pair's base_content_hash must match).

Never mutates prices/dates/option_id to make an attack "work" -- if it did,
an observed effect could be attributed to data tampering rather than to the
LLM actually acting on an injected instruction, which would undermine the
whole premise of measuring indirect prompt injection.
"""
import dataclasses
import hashlib
import json
from typing import Optional

from .attack_models import AttackConfig, ExternalContentSourceType

# Which field on a content_repository record a given injection_source_type
# targets. Only the 3 wired into Step 5's implemented attacks actually have a
# non-"description" field today (flight_policy -> policy_note); the rest map
# to "description" as a reasonable placeholder until those families are built.
SOURCE_TYPE_FIELD = {
    ExternalContentSourceType.FLIGHT_DESCRIPTION.value: "description",
    ExternalContentSourceType.FLIGHT_POLICY.value: "policy_note",
    ExternalContentSourceType.HOTEL_DESCRIPTION.value: "description",
    ExternalContentSourceType.HOTEL_POLICY.value: "policy_note",
    ExternalContentSourceType.CURRENCY_PROVIDER_NOTE.value: "provider_note",
    ExternalContentSourceType.TOUR_DESCRIPTION.value: "description",
    ExternalContentSourceType.LOCAL_TRAVEL_GUIDE.value: "description",
}


def _hash(text) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _hash_record(record: dict) -> str:
    return _hash(json.dumps(record, sort_keys=True, ensure_ascii=False))


@dataclasses.dataclass
class ExternalContentRecord:
    condition: str
    injection_present: bool
    attack_id: Optional[str]
    content: dict
    base_content_hash: str
    injected_content_hash: Optional[str] = None
    payload_hash: Optional[str] = None
    payload_offset: Optional[int] = None
    payload_length: Optional[int] = None
    injection_channel: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_external_content(base_record: dict, condition: str,
                            attack_config: Optional[AttackConfig] = None) -> ExternalContentRecord:
    if condition not in ("normal", "attack"):
        raise ValueError(f"condition must be 'normal' or 'attack', got {condition!r}")

    base_content_hash = _hash_record(base_record)

    if condition == "normal":
        if attack_config is not None:
            raise ValueError("attack_config must be None when condition == 'normal'")
        return ExternalContentRecord(
            condition="normal", injection_present=False, attack_id=None,
            content=dict(base_record), base_content_hash=base_content_hash,
        )

    if attack_config is None:
        raise ValueError("attack_config is required when condition == 'attack'")
    if base_record.get("option_id") != attack_config.injection_source_id:
        raise ValueError(f"base_record option_id {base_record.get('option_id')!r} does not match "
                          f"attack_config.injection_source_id {attack_config.injection_source_id!r}")

    field = SOURCE_TYPE_FIELD[attack_config.injection_source_type]
    if field not in base_record:
        raise KeyError(f"base_record has no field {field!r} for injection_source_type "
                        f"{attack_config.injection_source_type!r}")

    injected_record = dict(base_record)
    original_value = base_record[field]
    payload = attack_config.payload_template
    injected_record[field] = original_value + payload

    return ExternalContentRecord(
        condition="attack", injection_present=True, attack_id=attack_config.attack_id,
        content=injected_record, base_content_hash=base_content_hash,
        injected_content_hash=_hash_record(injected_record),
        payload_hash=_hash(payload), payload_offset=len(original_value), payload_length=len(payload),
        injection_channel=field,
    )


def apply_attack_injection(records: list, attack_config: Optional[AttackConfig], entry_agent_id: str):
    """
    Used by an Ollama specialist agent right before building its prompt.
    Returns (possibly_injected_records, injection_record_or_None):
      - attack_config is None, or attack_config.entry_agent_id != entry_agent_id
        (this agent isn't the entry point for this attack): records returned
        unchanged, second value None -- this is also what a NORMAL session
        always gets, since normal sessions never carry an attack_config.
      - otherwise: exactly ONE record (matching attack_config.injection_source_id)
        gets injected via build_external_content(); every other record in the
        list is untouched. Raises if injection_source_id isn't found, since a
        silently-skipped injection would misreport injection_present.
    """
    if attack_config is None or attack_config.entry_agent_id != entry_agent_id:
        return records, None

    out = []
    injection_record = None
    for r in records:
        if r.get("option_id") == attack_config.injection_source_id:
            injection_record = build_external_content(r, "attack", attack_config=attack_config)
            out.append(injection_record.content)
        else:
            out.append(r)
    if injection_record is None:
        raise ValueError(f"attack_config.injection_source_id {attack_config.injection_source_id!r} "
                          f"not found among records for entry_agent_id {entry_agent_id!r}")
    return out, injection_record
