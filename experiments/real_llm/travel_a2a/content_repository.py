"""
[Step 3-3] External content repository -- loads the source-controlled travel
content fixtures (data/travel_a2a/development/content/{flights,hotels,currency,tours}.json)
and provides lookup helpers. No real API/internet calls -- every record here
is a hand-authored fixture, internally consistent within itself (destination/
date/price arithmetic) but not claiming real-world accuracy.

[Step 6.5-1] This DEFAULT_CONTENT_DIR is the DEVELOPMENT content bundle only
-- formal_workload/ content (Step 6.5 onward) lives under a separate path and
is loaded through its own loader, never through this module's default.

Every record is convertible to a Part via content_record_to_part() below,
always with injection_present=False / attack_id=None -- this repository has
no attack-insertion capability itself (per the Step 3 scope boundary); base
content and diagnostic metadata are kept separate so a future attack-injection
step can wrap/extend a record without touching this loader.
"""
import json
import os
from typing import List, Optional

from .models import Part, PartType, SourceType

DEFAULT_CONTENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "data", "travel_a2a", "development", "content")


class ContentRepository:
    def __init__(self, flights: List[dict], hotels: List[dict], currency: List[dict], tours: List[dict]):
        self._flights = flights
        self._hotels = hotels
        self._tours = tours
        self._currency_by_pair = {c["pair"]: c for c in currency}

    def flights_for(self, destination: str) -> List[dict]:
        return [f for f in self._flights if f["destination"] == destination]

    def hotels_for(self, destination: str) -> List[dict]:
        return [h for h in self._hotels if h["destination"] == destination]

    def tours_for(self, destination: str) -> List[dict]:
        return [t for t in self._tours if t["destination"] == destination]

    def tours_for_in_range(self, destination: str, start_date: str, end_date: str) -> List[dict]:
        """start_date/end_date: 'YYYY-MM-DD' strings (ISO date, lexicographically
        comparable). Used by MockToursAgent's initial delivery pass -- a
        destination whose fixture tours all fall OUTSIDE the requested trip
        window returns [] here, which is exactly what drives the
        integration_revision branch (Step 3-6.D) for hard_multi_constraint_london."""
        return [t for t in self.tours_for(destination) if start_date <= t["date"] <= end_date]

    def currency_rate(self, base_currency: str, target_currency: str) -> float:
        pair = f"{base_currency}/{target_currency}"
        entry = self._currency_by_pair.get(pair)
        if entry is None:
            raise KeyError(f"no currency rate fixture for pair {pair!r}")
        return entry["rate"]

    def currency_record(self, base_currency: str, target_currency: str) -> dict:
        pair = f"{base_currency}/{target_currency}"
        entry = self._currency_by_pair.get(pair)
        if entry is None:
            raise KeyError(f"no currency rate fixture for pair {pair!r}")
        return entry


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_content_repository(base_dir: str = DEFAULT_CONTENT_DIR) -> ContentRepository:
    return ContentRepository(
        flights=_load_json(os.path.join(base_dir, "flights.json")),
        hotels=_load_json(os.path.join(base_dir, "hotels.json")),
        currency=_load_json(os.path.join(base_dir, "currency.json")),
        tours=_load_json(os.path.join(base_dir, "tours.json")),
    )


def content_record_to_part(record: dict, part_id: str, source_type: SourceType = SourceType.EXTERNAL_CONTENT,
                            created_at: str = "", source_id: Optional[str] = None) -> Part:
    """Wraps one raw content-repository record (a flight/hotel/tour option
    dict, or a currency-rate dict) as a Part. Always injection_present=False /
    attack_id=None here -- this repository is base content only."""
    return Part(
        part_id=part_id, part_type=PartType.DATA, mime_type="application/json",
        content=record, source_type=source_type, source_id=source_id,
        injection_present=False, attack_id=None, created_at=created_at,
    )
