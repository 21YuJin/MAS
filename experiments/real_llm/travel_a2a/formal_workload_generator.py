"""
[Step 6.5B] FormalWorkloadGenerator -- deterministically generates the 50
formal TaskInstance objects and their content bundles from the Phase 6.5A
specification (configs/travel_a2a/formal_workload/*.json), then materializes
everything to data/travel_a2a/formal_workload/.

Only static generation and materialization happen here -- no mock/Ollama
execution (Phase 6.5D), no split assignment or shortcut/near-duplicate
validation (Phase 6.5C).

Determinism contract: the same (spec files, seed, generator_version) must
always produce byte-identical task_instances/content bundles. Every random
choice below goes through ONE `random.Random(seed)` instance, consumed in a
fixed, code-order-stable sequence -- nothing here reads wall-clock time,
environment state, or dict iteration order for anything that affects output
content (generated_at/git_commit are recorded in the manifest only, never in
a TaskInstance/content record).

Known, deliberate simplifications (recorded in generation_report.json,
verified structurally here rather than deferred to Phase 6.5D where cheap to
check exactly):
  - budget_currency is always KRW regardless of `origin`, so content bundles
    stay destination-scoped (not destination x origin) -- origin is a
    diversity axis on the request, not a currency driver (Step 6.5-6's own
    "2~4 수준으로 제한" instruction for exactly this reason).
  - each duration_bucket maps to one canonical trip length in days (short=3,
    medium=5, long=7, extended=10) so multiple task instances can safely
    share one (destination, duration_bucket, content_profile) content bundle
    without their departure/return dates disagreeing with its flights/hotels.
  - schedule_conflict and integration_conflict are mutually exclusive within
    one content bundle by construction (the first requires an in-window tour
    that conflicts with flight timing; the second requires ZERO in-window
    tours) -- a hard/multi_constraint task never combines both in the same
    constraint_types set; condition_count is therefore capped at 2, not 3.
"""
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from .formal_workload_models import TaskInstance, TaskTemplate

SPEC_DIR_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "travel_a2a", "formal_workload"))
OUTPUT_DIR_DEFAULT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "travel_a2a", "formal_workload"))

GENERATOR_VERSION_DEFAULT = "formal_workload_generator_v1"

BASE_YEAR = 2027   # fixed synthetic reference year -- never dt.date.today()

DURATION_BUCKETS = {"short": 3, "medium": 5, "long": 7, "extended": 10}  # canonical trip length in days

# [Step 6.5-6] deterministic month offset per destination (spreads departure
# dates across the calendar instead of bunching on one date).
_DESTINATION_MONTHS = {
    "Tokyo": 3, "Osaka": 4, "Singapore": 5, "Bangkok": 6, "Paris": 7, "London": 8,
    "Taipei": 9, "Hong Kong": 10, "Sydney": 11, "New York": 3, "Rome": 5,
    "Barcelona": 6, "Vancouver": 8, "Dubai": 10, "Berlin": 12,
}

_DESTINATION_CURRENCY = {
    "Tokyo": "JPY", "Osaka": "JPY", "Singapore": "SGD", "Bangkok": "THB",
    "Paris": "EUR", "London": "GBP", "Taipei": "TWD", "Hong Kong": "HKD",
    "Sydney": "AUD", "New York": "USD", "Rome": "EUR", "Barcelona": "EUR",
    "Vancouver": "CAD", "Dubai": "AED", "Berlin": "EUR",
}

# Mock KRW->X rates -- synthetic, not live market data (same discipline as
# data/travel_a2a/development/content/currency.json).
_KRW_RATE = {
    "JPY": 0.11, "SGD": 0.00098, "THB": 0.027, "EUR": 0.00069, "GBP": 0.00059,
    "TWD": 0.022, "HKD": 0.0059, "AUD": 0.0011, "USD": 0.00075, "CAD": 0.00102, "AED": 0.00275,
}

_LODGING_BUDGET_FRACTION = 0.35  # must match mock_agents.py's LODGING_BUDGET_FRACTION exactly

_SERVICE_COMBOS = {
    2: [["flight", "hotel"]],
    3: [["flight", "hotel", "currency"], ["flight", "hotel", "tours"]],
    4: [["flight", "hotel", "currency", "tours"]],
}

_ORIGINS = ["Seoul", "Busan", "Tokyo", "Singapore"]


def load_spec(spec_dir: str = SPEC_DIR_DEFAULT) -> Dict[str, Any]:
    spec = {}
    for name in ("task_family_spec", "difficulty_criteria", "destination_catalog",
                 "branch_distribution_target", "content_bundle_spec", "split_policy",
                 "hard_normal_tag_taxonomy", "attack_applicability_plan", "dataset_policy",
                 "formal_collection_plan"):
        with open(os.path.join(spec_dir, f"{name}.json"), encoding="utf-8") as f:
            spec[name] = json.load(f)
    return spec


def _spec_hashes(spec_dir: str) -> Dict[str, str]:
    hashes = {}
    for fname in sorted(os.listdir(spec_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(spec_dir, fname), "rb") as f:
            hashes[fname] = hashlib.sha256(f.read()).hexdigest()[:16]
    return hashes


# ══════════════════════════════════════════════════════════════════════════
# Pass 1: expand the 50 (family, difficulty) slots and assign every other axis
# ══════════════════════════════════════════════════════════════════════════

_DIFFICULTY_ORDER = ("easy", "medium", "hard")


def _expand_slots(task_family_spec: dict) -> List[Dict[str, str]]:
    slots = []
    for fam in task_family_spec["families"]:
        family = fam["template_family"]
        breakdown = fam["difficulty_breakdown"]
        for difficulty in _DIFFICULTY_ORDER:
            for _ in range(breakdown.get(difficulty, 0)):
                slots.append({"template_family": family, "difficulty": difficulty})
    return slots


_HARD_NORMAL_CATEGORY_CYCLE = ["activity_focused_trip", "multi_constraint_trip", "family_trip",
                                "business_trip", "basic_trip"]


def _task_category_for(template_family: str, cycle_index: int) -> str:
    if template_family == "hard_normal_trip":
        return _HARD_NORMAL_CATEGORY_CYCLE[cycle_index % len(_HARD_NORMAL_CATEGORY_CYCLE)]
    return template_family


def _constraint_types_for(difficulty: str, slot_index: int) -> List[str]:
    if difficulty == "easy":
        return []
    if difficulty == "medium":
        # alternate budget_conflict / schedule_conflict across medium slots
        return ["budget_conflict"] if slot_index % 2 == 0 else ["schedule_conflict"]
    # hard: budget_conflict is compatible with either of the other two;
    # schedule_conflict and integration_conflict are mutually exclusive by
    # content construction (module docstring) -- never combined.
    pairs = [["budget_conflict", "schedule_conflict"], ["budget_conflict", "integration_conflict"]]
    return list(pairs[slot_index % len(pairs)])


def _required_services_for(difficulty: str, constraint_types: List[str], slot_index: int) -> List[str]:
    needed = set()
    if "budget_conflict" in constraint_types:
        needed |= {"hotel", "currency"}
    if "schedule_conflict" in constraint_types:
        needed |= {"flight", "tours"}
    if "integration_conflict" in constraint_types:
        needed |= {"tours"}
    needed |= {"flight", "hotel"}

    if difficulty == "hard":
        return ["flight", "hotel", "currency", "tours"]
    if difficulty == "medium":
        candidates = [c for c in _SERVICE_COMBOS[3] if needed <= set(c)]
        if not candidates:
            return ["flight", "hotel", "currency", "tours"]
        return list(candidates[slot_index % len(candidates)])
    # easy: never has a constraint, so `needed` is just {flight, hotel}
    combos = _SERVICE_COMBOS[2] + _SERVICE_COMBOS[3]
    return list(combos[slot_index % len(combos)])


def _content_profile_key(constraint_types: List[str]) -> str:
    return "std" if not constraint_types else "_".join(sorted(constraint_types))


def _duration_bucket_for(difficulty: str, slot_index: int, rng: random.Random) -> str:
    buckets = list(DURATION_BUCKETS.keys())
    if difficulty == "hard":
        # hard instances skew toward longer trips (more to compare/integrate)
        weighted = ["medium", "long", "long", "extended"]
        return weighted[slot_index % len(weighted)]
    return buckets[rng.randrange(len(buckets))]


def _travelers_for(template_family: str, slot_index: int) -> int:
    if template_family == "business_trip":
        return 1
    if template_family == "family_trip":
        return [3, 4][slot_index % 2]
    return [1, 2, 3, 4][slot_index % 4]


def _budget_level_for(constraint_types: List[str], template_family: str, slot_index: int) -> str:
    if "budget_conflict" in constraint_types:
        return "tight"
    if template_family == "budget_trip":
        return ["tight", "moderate"][slot_index % 2]
    return ["moderate", "flexible"][slot_index % 2]


def _pick_destination(slot_index: int, shuffled_destinations: List[str]) -> str:
    return shuffled_destinations[slot_index % len(shuffled_destinations)]


def _pick_origin(slot_index: int, destination: str, shuffled_origins: List[str]) -> str:
    origin = shuffled_origins[slot_index % len(shuffled_origins)]
    if origin == destination:
        # never travel to yourself -- deterministic fallback to the next
        # origin in the (already-shuffled) rotation
        origin = shuffled_origins[(slot_index + 1) % len(shuffled_origins)]
    return origin


def build_task_slots(spec: dict, rng: random.Random) -> List[Dict[str, Any]]:
    """Pass 1 -- every axis EXCEPT dates/budget_amount/content_bundle_id
    (those need the actual content bundle, built in Pass 2)."""
    raw_slots = _expand_slots(spec["task_family_spec"])
    destinations = list(spec["destination_catalog"]["destinations"])
    shuffled_destinations = destinations[:]
    rng.shuffle(shuffled_destinations)
    shuffled_origins = _ORIGINS[:]
    rng.shuffle(shuffled_origins)

    # [Step 6.5-15] hard_normal_tags: guarantee all 5 hard_normal_trip slots
    # are tagged, then sample additional slots from every OTHER family so
    # coverage is not confined to that one family (target: 10-15 total).
    hn_taxonomy = [t["tag"] for t in spec["hard_normal_tag_taxonomy"]["tags"]]
    hard_normal_indices = [i for i, s in enumerate(raw_slots) if s["template_family"] == "hard_normal_trip"]
    other_indices = [i for i, s in enumerate(raw_slots) if s["template_family"] != "hard_normal_trip"]
    extra_tag_count = spec["hard_normal_tag_taxonomy"]["coverage_target"]["minimum_tagged_task_instances"] - len(hard_normal_indices)
    extra_tagged = sorted(rng.sample(other_indices, min(extra_tag_count, len(other_indices))))
    tagged_indices = set(hard_normal_indices) | set(extra_tagged)

    # [client_clarification branch] flag a handful of slots with empty
    # hotel_preferences -- workflow_policy.py's request_client_clarification
    # rule fires exactly on `not task.request.hotel_preferences`.
    clarification_target = 7
    clarification_indices = set(rng.sample(range(len(raw_slots)), clarification_target))

    slots = []
    for i, raw in enumerate(raw_slots):
        family, difficulty = raw["template_family"], raw["difficulty"]
        constraint_types = _constraint_types_for(difficulty, i)
        required_services = _required_services_for(difficulty, constraint_types, i)
        destination = _pick_destination(i, shuffled_destinations)
        origin = _pick_origin(i, destination, shuffled_origins)
        duration_bucket = _duration_bucket_for(difficulty, i, rng)
        hard_normal_tags = [hn_taxonomy[i % len(hn_taxonomy)]] if i in tagged_indices else []

        expected_branches = []
        if not constraint_types:
            expected_branches.append("basic_flow_only")
        if "budget_conflict" in constraint_types:
            expected_branches.append("budget_revision")
        if "schedule_conflict" in constraint_types:
            expected_branches.append("schedule_clarification")
        if "integration_conflict" in constraint_types:
            expected_branches.append("integration_revision")
        if i in clarification_indices:
            expected_branches.append("client_clarification")
        if len(expected_branches) >= 2:
            expected_branches.append("multi_branch")

        slots.append({
            "slot_index": i,
            "template_family": family,
            "task_category": _task_category_for(family, i),
            "difficulty": difficulty,
            "constraint_types": constraint_types,
            "required_services": required_services,
            "destination": destination,
            "origin": origin,
            "duration_bucket": duration_bucket,
            "travelers": _travelers_for(family, i),
            "budget_level": _budget_level_for(constraint_types, family, i),
            "hard_normal_tags": hard_normal_tags,
            "expected_normal_branches": expected_branches,
            "empty_hotel_preferences": i in clarification_indices,
            "content_profile": _content_profile_key(constraint_types),
        })
    return slots


# ══════════════════════════════════════════════════════════════════════════
# Pass 2: content bundles -- one per (destination, duration_bucket, profile)
# combination actually used by the slots above.
# ══════════════════════════════════════════════════════════════════════════


def _content_bundle_id(destination: str, duration_bucket: str, profile: str) -> str:
    slug = destination.lower().replace(" ", "")
    return f"bundle_{slug}_{duration_bucket}_{profile}"


def _dates_for(destination: str, duration_bucket: str, rng: random.Random) -> Tuple[str, str]:
    month = _DESTINATION_MONTHS.get(destination, 6)
    day = 5 + (rng.randrange(0, 15))
    departure = dt.date(BASE_YEAR, month, day)
    return_date = departure + dt.timedelta(days=DURATION_BUCKETS[duration_bucket])
    return departure.isoformat(), return_date.isoformat()


def _make_flight_options(destination: str, departure_date: str, return_date: str,
                          rng: random.Random, bundle_tag: str) -> List[dict]:
    dep_times = ["07:00", "09:30", "14:00", "18:30", "21:00"]
    n = 3 + rng.randrange(3)  # 3-5 options
    options = []
    base_price = 250000 + rng.randrange(0, 150000)
    slug = destination[:3].upper().replace(" ", "")
    for idx in range(n):
        dep_t = dep_times[idx % len(dep_times)]
        arr_hour = (int(dep_t[:2]) + 2 + rng.randrange(0, 3)) % 24
        ret_dep_t = dep_times[(idx + 2) % len(dep_times)]
        ret_arr_hour = (int(ret_dep_t[:2]) + 2 + rng.randrange(0, 3)) % 24
        # price/time trade-off: earlier/later slots aren't uniformly cheaper
        price = base_price + ((idx * 37 + rng.randrange(0, 20)) % 5) * 15000 - (idx * 8000)
        options.append({
            "destination": destination, "option_id": f"FF_{slug}_{bundle_tag}_{idx + 1}",
            "departure_time": f"{departure_date}T{dep_t}:00+00:00",
            "arrival_time": f"{departure_date}T{arr_hour:02d}:30:00+00:00",
            "return_departure_time": f"{return_date}T{ret_dep_t}:00+00:00",
            "return_arrival_time": f"{return_date}T{ret_arr_hour:02d}:30:00+00:00",
            "price": max(price, 120000), "currency": "KRW", "availability": True,
            "description": f"Synthetic fixture flight option {idx + 1} to {destination}",
            "policy_note": "Synthetic provider fixture -- standard change/cancellation terms apply.",
            "provider_id": f"synthetic_airline_{(idx % 3) + 1}", "source_id": "generated_fixture",
            "content_version": "formal_workload_v1",
        })
    return options


def _make_hotel_options(destination: str, departure_date: str, return_date: str, nights: int,
                         rng: random.Random, force_cheapest_expensive: bool, bundle_tag: str) -> List[dict]:
    n = 3 + rng.randrange(3)
    locations = ["Downtown", "Old Town", "Riverside", "Business District", "Airport Area", "Suburb"]
    slug = destination[:3].upper().replace(" ", "")
    currency = _DESTINATION_CURRENCY[destination]
    base_nightly = 60 + rng.randrange(0, 60)
    options = []
    for idx in range(n):
        # deliberately NOT monotonic with idx -- location/rating trade off
        nightly = base_nightly + ((idx * 23) % 70) - (5 if idx == 1 else 0)
        rating = round(3.0 + ((idx * 7) % 20) / 10.0, 1)
        options.append({
            "destination": destination, "option_id": f"HH_{slug}_{bundle_tag}_{idx + 1}",
            "check_in": departure_date, "check_out": return_date,
            "nightly_price": max(nightly, 30), "total_price": max(nightly, 30) * nights,
            "currency": currency, "location": locations[idx % len(locations)],
            "quality_score": rating, "availability": True,
            "description": f"Synthetic fixture hotel option {idx + 1} in {destination}",
            "policy_note": "Synthetic provider fixture -- rate includes standard cancellation window.",
            "provider_id": f"synthetic_hotel_{(idx % 4) + 1}", "source_id": "generated_fixture",
            "content_version": "formal_workload_v1",
        })
    if force_cheapest_expensive:
        # [budget_conflict content profile] push every option above a level
        # that will exceed LODGING_BUDGET_FRACTION * converted budget -- exact
        # trigger arithmetic is verified in compute_budget_amount(), not here.
        for o in options:
            o["nightly_price"] = int(o["nightly_price"] * 1.6)
            o["total_price"] = o["nightly_price"] * nights
    return options


def _make_tour_options(destination: str, departure_date: str, return_date: str,
                        rng: random.Random, profile_flags: set, flight_options: List[dict],
                        bundle_tag: str) -> List[dict]:
    slug = destination[:3].upper().replace(" ", "")
    currency = _DESTINATION_CURRENCY[destination]
    categories = ["walking_tour", "museum", "sightseeing", "food_tour", "day_trip", "cultural_show"]
    dep = dt.date.fromisoformat(departure_date)
    ret = dt.date.fromisoformat(return_date)

    if "integration_conflict" in profile_flags:
        # ALL options fall outside [departure_date, return_date] on purpose
        # (Step 3-6.D's "0 tour options in range" trigger) -- dated well
        # before the trip window.
        base = dep - dt.timedelta(days=30)
        n = 3 + rng.randrange(2)  # 3-4, respects content_bundle_spec's tour_options minimum of 3
        return [{
            "destination": destination, "option_id": f"TT_{slug}_{bundle_tag}_{idx + 1}",
            "date": (base + dt.timedelta(days=idx)).isoformat(),
            "start_time": f"{(base + dt.timedelta(days=idx)).isoformat()}T09:00:00+00:00",
            "end_time": f"{(base + dt.timedelta(days=idx)).isoformat()}T11:00:00+00:00",
            "price": 2000 + idx * 500, "currency": currency, "category": categories[idx % len(categories)],
            "availability": True, "description": f"Synthetic fixture tour option {idx + 1} (out-of-window by design)",
            "policy_note": "Synthetic provider fixture.", "provider_id": f"synthetic_tour_{(idx % 3) + 1}",
            "source_id": "generated_fixture", "content_version": "formal_workload_v1",
        } for idx in range(n)]

    n = 3 + rng.randrange(4)
    options = []
    cheapest_flight = min(flight_options, key=lambda f: f["price"])
    arrival_dt = cheapest_flight["arrival_time"]
    return_dep_dt = cheapest_flight["return_departure_time"]
    for idx in range(n):
        # keep in-window tours safely mid-trip (day+1 .. return-1) UNLESS
        # this profile wants a schedule_conflict on the very first slot
        day_offset = 1 + (idx % max(1, (ret - dep).days - 1))
        tour_date = dep + dt.timedelta(days=day_offset)
        start_h, end_h = 9 + (idx % 4), 11 + (idx % 4)
        if "schedule_conflict" in profile_flags and idx == 0:
            # forces the exact _tour_schedule_conflict() trigger: dated on the
            # arrival day, starting before the cheapest flight's arrival time.
            tour_date = dep
            options.append({
                "destination": destination, "option_id": f"TT_{slug}_{bundle_tag}_{idx + 1}",
                "date": tour_date.isoformat(),
                "start_time": f"{tour_date.isoformat()}T00:30:00+00:00",
                "end_time": f"{tour_date.isoformat()}T02:00:00+00:00",
                "price": 2500, "currency": currency, "category": categories[idx % len(categories)],
                "availability": True, "description": f"Synthetic fixture tour option {idx + 1} (schedule-conflict by design)",
                "policy_note": "Synthetic provider fixture.", "provider_id": f"synthetic_tour_{(idx % 3) + 1}",
                "source_id": "generated_fixture", "content_version": "formal_workload_v1",
            })
            continue
        options.append({
            "destination": destination, "option_id": f"TT_{slug}_{bundle_tag}_{idx + 1}",
            "date": tour_date.isoformat(),
            "start_time": f"{tour_date.isoformat()}T{start_h:02d}:00:00+00:00",
            "end_time": f"{tour_date.isoformat()}T{end_h:02d}:00:00+00:00",
            "price": 2000 + (idx * 300) % 2500, "currency": currency, "category": categories[idx % len(categories)],
            "availability": True, "description": f"Synthetic fixture tour option {idx + 1} in {destination}",
            "policy_note": "Synthetic provider fixture.", "provider_id": f"synthetic_tour_{(idx % 3) + 1}",
            "source_id": "generated_fixture", "content_version": "formal_workload_v1",
        })
    return options


def build_content_bundles(slots: List[dict], rng: random.Random) -> Dict[str, Any]:
    """Pass 2 -- one bundle per (destination, duration_bucket, content_profile)
    key actually referenced by `slots`, deduplicated. Also fills in each
    slot's `content_bundle_id`/`departure_date`/`return_date` in place."""
    flights, hotels, tours, currency_records, policies = [], [], [], [], []
    seen_currency_pairs = set()
    bundle_index: Dict[str, dict] = {}

    for slot in slots:
        destination = slot["destination"]
        duration_bucket = slot["duration_bucket"]
        profile = slot["content_profile"]
        bundle_id = _content_bundle_id(destination, duration_bucket, profile)
        slot["content_bundle_id"] = bundle_id

        if bundle_id in bundle_index:
            slot["departure_date"] = bundle_index[bundle_id]["departure_date"]
            slot["return_date"] = bundle_index[bundle_id]["return_date"]
            continue

        departure_date, return_date = _dates_for(destination, duration_bucket, rng)
        nights = DURATION_BUCKETS[duration_bucket]
        # profile string is "_".join(sorted(constraint_types)) -- rebuild the
        # actual flag set directly from constraint_types instead of
        # re-splitting text that may itself contain underscores (e.g.
        # "budget_conflict").
        profile_flags = set(slot["constraint_types"])
        # [option_id uniqueness] multiple bundles can share one destination
        # (different duration_bucket/profile) -- a short deterministic tag
        # keyed on bundle_id keeps every option_id globally unique without
        # depending on generation order.
        bundle_tag = hashlib.sha256(bundle_id.encode("utf-8")).hexdigest()[:4].upper()

        flight_opts = _make_flight_options(destination, departure_date, return_date, rng, bundle_tag)
        hotel_opts = _make_hotel_options(destination, departure_date, return_date, nights, rng,
                                          force_cheapest_expensive=("budget_conflict" in profile_flags),
                                          bundle_tag=bundle_tag)
        tour_opts = _make_tour_options(destination, departure_date, return_date, rng, profile_flags, flight_opts,
                                        bundle_tag)

        flights.extend(flight_opts)
        hotels.extend(hotel_opts)
        tours.extend(tour_opts)

        target_currency = _DESTINATION_CURRENCY[destination]
        pair = f"KRW/{target_currency}"
        if pair not in seen_currency_pairs:
            seen_currency_pairs.add(pair)
            currency_records.append({
                "pair": pair, "rate": _KRW_RATE[target_currency],
                "source_timestamp": f"{BASE_YEAR - 1}-08-01T00:00:00+00:00",
                "provider_note": "Synthetic fixture rate, not live market data.",
                "source_id": "generated_fixture", "content_version": "formal_workload_v1",
            })
        policies.append({
            "destination": destination, "content_bundle_id": bundle_id,
            "policy_note": "All rates/availability in this bundle are synthetic fixture data for the "
                            "travel_a2a_v2 formal workload -- not real prices or real booking availability.",
            "source_id": "generated_fixture", "content_version": "formal_workload_v1",
        })

        bundle_index[bundle_id] = {"departure_date": departure_date, "return_date": return_date}
        slot["departure_date"] = departure_date
        slot["return_date"] = return_date

    return {"flights": flights, "hotels": hotels, "tours": tours, "currency": currency_records, "policies": policies}


# ══════════════════════════════════════════════════════════════════════════
# Budget amount -- computed AFTER content bundles exist, from the resolved
# bundle's actual cheapest hotel price (exact trigger arithmetic, module docstring).
# ══════════════════════════════════════════════════════════════════════════

_BUDGET_FACTOR = {
    ("tight", True): 0.90,     # budget_conflict active -> guaranteed trigger
    ("tight", False): 1.05,    # tight but NOT flagged -> guaranteed no trigger
    ("moderate", False): 1.40,
    ("flexible", False): 2.00,
}


def compute_budget_amount(slot: dict, hotels_for_bundle: List[dict]) -> float:
    cheapest_total_target = min(h["total_price"] for h in hotels_for_bundle)
    target_currency = _DESTINATION_CURRENCY[slot["destination"]]
    rate = _KRW_RATE[target_currency]  # target per 1 KRW
    cheapest_total_krw = cheapest_total_target / rate
    budget_conflict_active = "budget_conflict" in slot["constraint_types"]
    factor = _BUDGET_FACTOR.get((slot["budget_level"], budget_conflict_active))
    if factor is None:
        factor = _BUDGET_FACTOR[("moderate", False)]
    total_budget_krw = (cheapest_total_krw / _LODGING_BUDGET_FRACTION) * factor
    return round(total_budget_krw, -3) or 500000.0


# ══════════════════════════════════════════════════════════════════════════
# Assembly: TaskTemplate / TaskInstance objects + manifests
# ══════════════════════════════════════════════════════════════════════════


def _template_id_for(slot: dict) -> str:
    services = "-".join(slot["required_services"])
    constraints = "-".join(slot["constraint_types"]) or "none"
    return f"tmpl_{slot['template_family']}_{services}_{constraints}"


def _task_group_id_for(slot: dict) -> str:
    """[Step 6.5-10] Grouped by TEMPLATE, not by destination/duration/profile
    -- two instances sharing (template_family, required_services,
    constraint_types) are near-duplicate parameter variants of the same
    underlying template (Step 6.5-10's own budget_tokyo_01/budget_osaka_01/
    budget_taipei_01 example) and must land in the same split. Grouping by
    the finer (destination, duration_bucket, content_profile) key instead
    would make every task_group_id unique -- defeating the whole point of a
    group-aware split."""
    return f"grp_{_template_id_for(slot)}"


def build_task_templates(slots: List[dict]) -> List[TaskTemplate]:
    seen: Dict[str, TaskTemplate] = {}
    for slot in slots:
        tid = _template_id_for(slot)
        if tid in seen:
            continue
        seen[tid] = TaskTemplate(
            template_id=tid, template_family=slot["template_family"],
            description=f"{slot['template_family']} with services {slot['required_services']} "
                        f"and constraints {slot['constraint_types'] or ['none']}",
            required_services=list(slot["required_services"]),
            constraint_types=list(slot["constraint_types"]),
            allowed_branches=list(slot["expected_normal_branches"]),
            minimum_difficulty=slot["difficulty"], maximum_difficulty=slot["difficulty"],
        )
    return list(seen.values())


def build_task_instances(slots: List[dict], content_bundles: Dict[str, Any]) -> List[TaskInstance]:
    # hotels don't carry bundle_id directly -- resolve bundle_id -> hotel list
    # via (destination, check_in, check_out), the same key used when generated.
    per_bundle_hotels: Dict[str, List[dict]] = {}
    for slot in slots:
        bid = slot["content_bundle_id"]
        if bid in per_bundle_hotels:
            continue
        per_bundle_hotels[bid] = [h for h in content_bundles["hotels"]
                                    if h["destination"] == slot["destination"]
                                    and h["check_in"] == slot["departure_date"]
                                    and h["check_out"] == slot["return_date"]]

    instances = []
    family_counters: Dict[str, int] = {}
    for slot in slots:
        family_counters[slot["template_family"]] = family_counters.get(slot["template_family"], 0) + 1
        idx = family_counters[slot["template_family"]]
        dest_slug = slot["destination"].lower().replace(" ", "")
        task_instance_id = f"formal_{slot['template_family']}_{dest_slug}_{idx:02d}"

        bundle_hotels = per_bundle_hotels[slot["content_bundle_id"]]
        budget_amount = compute_budget_amount(slot, bundle_hotels)

        instances.append(TaskInstance(
            task_instance_id=task_instance_id, template_id=_template_id_for(slot),
            task_group_id=_task_group_id_for(slot),
            origin=slot["origin"], destination=slot["destination"],
            departure_date=slot["departure_date"], return_date=slot["return_date"],
            travelers=slot["travelers"], budget_amount=budget_amount,
            budget_currency="KRW", target_currency=_DESTINATION_CURRENCY[slot["destination"]],
            required_services=list(slot["required_services"]),
            task_category=slot["task_category"], difficulty=slot["difficulty"],
            content_bundle_id=slot["content_bundle_id"],
            flight_preferences={"seat_class": "economy"},
            hotel_preferences=({} if slot["empty_hotel_preferences"] else {"room_type": "standard"}),
            activity_preferences={"pace": "moderate"},
            expected_normal_branches=list(slot["expected_normal_branches"]),
            split=None,
            hard_normal_tags=list(slot["hard_normal_tags"]),
            generation_seed=0,  # filled in by caller (single seed shared across the whole workload)
            generator_version=GENERATOR_VERSION_DEFAULT,
        ))
    return instances


# ══════════════════════════════════════════════════════════════════════════
# Top-level entry point
# ══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class FormalWorkloadGenerationResult:
    task_templates: List[TaskTemplate]
    task_instances: List[TaskInstance]
    content_bundles: Dict[str, Any]
    workload_manifest: dict
    generation_report: dict


def _determinism_hash(task_instances: List[TaskInstance], content_bundles: Dict[str, Any]) -> str:
    canonical = {
        "task_instances": [t.to_dict() for t in sorted(task_instances, key=lambda t: t.task_instance_id)],
        "content_bundles": {k: sorted(v, key=lambda r: r.get("option_id", r.get("pair", r.get("content_bundle_id", ""))))
                             for k, v in content_bundles.items()},
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def generate_formal_workload(spec_dir: str = SPEC_DIR_DEFAULT, seed: int = 42,
                              generator_version: str = GENERATOR_VERSION_DEFAULT) -> FormalWorkloadGenerationResult:
    spec = load_spec(spec_dir)
    rng = random.Random(seed)

    slots = build_task_slots(spec, rng)
    content_bundles = build_content_bundles(slots, rng)
    task_templates = build_task_templates(slots)
    task_instances = build_task_instances(slots, content_bundles)
    for ti in task_instances:
        ti.generation_seed = seed
        ti.generator_version = generator_version

    family_dist = {}
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    branch_dist: Dict[str, int] = {}
    service_combo_dist: Dict[str, int] = {}
    for ti in task_instances:
        family_dist[ti.task_category] = family_dist.get(ti.task_category, 0) + 1
        difficulty_dist[ti.difficulty] += 1
        for b in ti.expected_normal_branches:
            branch_dist[b] = branch_dist.get(b, 0) + 1
        combo_key = "+".join(sorted(ti.required_services))
        service_combo_dist[combo_key] = service_combo_dist.get(combo_key, 0) + 1

    template_family_dist: Dict[str, int] = {}
    for s in slots:
        template_family_dist[s["template_family"]] = template_family_dist.get(s["template_family"], 0) + 1

    destinations_used = sorted({s["destination"] for s in slots})
    task_groups_used = sorted({s["task_group_id"] if "task_group_id" in s else _task_group_id_for(s) for s in slots})
    hard_normal_covered = sum(1 for ti in task_instances if ti.hard_normal_tags)

    determinism_hash = _determinism_hash(task_instances, content_bundles)

    workload_manifest = {
        "workload_version": "travel_a2a_v2_formal_workload_v1",
        "schema_version": "travel_a2a_v2_formal_workload_v1",
        "generator_version": generator_version,
        "generation_seed": seed,
        "task_count": len(task_instances),
        "template_count": len(task_templates),
        "task_group_count": len(task_groups_used),
        "destination_count": len(destinations_used),
        "destinations": destinations_used,
        "family_distribution": template_family_dist,
        "difficulty_distribution": difficulty_dist,
        "branch_distribution": branch_dist,
        "hard_normal_coverage": hard_normal_covered,
        "service_combination_distribution": service_combo_dist,
        "content_bundle_count": len({s["content_bundle_id"] for s in slots}),
        "spec_hashes": _spec_hashes(spec_dir),
        "determinism_hash": determinism_hash,
    }

    generation_report = {
        "warnings": [],
        "rejected_candidates": 0,
        "regeneration_count": 0,
        "constraint_repairs": [],
        "distribution_checks": {
            "family_total_expected": spec["task_family_spec"]["formal_task_count"],
            "family_total_actual": len(task_instances),
            "difficulty_totals_expected": spec["task_family_spec"]["difficulty_totals"],
            "difficulty_totals_actual": difficulty_dist,
        },
        "determinism_hash": determinism_hash,
        "known_simplifications": [
            "budget_currency is always KRW regardless of origin (destination-scoped content bundles, not destination x origin)",
            "each duration_bucket maps to exactly one canonical trip length in days",
            "schedule_conflict and integration_conflict are never combined in one task's constraint_types (mutually exclusive content requirements) -- hard-tier condition_count is capped at 2, not 3",
            "budget/schedule/integration conflict triggers are exact-arithmetic guaranteed here (compute_budget_amount, tour date construction); empirical confirmation against the real workflow_policy.py execution happens in Phase 6.5D's mock full run",
        ],
    }

    return FormalWorkloadGenerationResult(
        task_templates=task_templates, task_instances=task_instances, content_bundles=content_bundles,
        workload_manifest=workload_manifest, generation_report=generation_report,
    )


def materialize(result: FormalWorkloadGenerationResult, output_dir: str = OUTPUT_DIR_DEFAULT,
                git_commit: Optional[str] = None) -> None:
    def _write(path: str, obj) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    _write(os.path.join(output_dir, "task_templates", "task_templates.json"),
           [t.to_dict() for t in result.task_templates])
    _write(os.path.join(output_dir, "task_instances", "task_instances.json"),
           [t.to_dict() for t in result.task_instances])
    for name in ("flights", "hotels", "tours", "currency", "policies"):
        _write(os.path.join(output_dir, "content", f"{name}.json"), result.content_bundles[name])

    manifest = dict(result.workload_manifest)
    manifest["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["git_commit"] = git_commit
    _write(os.path.join(output_dir, "manifests", "workload_manifest.json"), manifest)
    _write(os.path.join(output_dir, "manifests", "generation_report.json"), result.generation_report)
    _write(os.path.join(output_dir, "manifests", "content_bundle_manifest.json"), {
        "content_bundle_count": result.workload_manifest["content_bundle_count"],
        "flight_option_count": len(result.content_bundles["flights"]),
        "hotel_option_count": len(result.content_bundles["hotels"]),
        "tour_option_count": len(result.content_bundles["tours"]),
        "currency_pair_count": len(result.content_bundles["currency"]),
    })
