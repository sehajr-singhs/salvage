"""The SQLite inventory store (brief 05) and the reverse-search query.

Tables ``devices``, ``parts``, ``verdicts`` (plus ``readings`` for
provenance) live in one SQLite file, ``data/inventory.db`` by default.
``add(device, parts, verdicts)`` remembers one scanned device;
``query(text) -> list[Match]`` answers requests like
``"I need a 12 V motor with a driver"``.

Query pipeline, per the brief:

1. ``parse_filters`` turns the request into structured filters with one
   cached LLM call (with a deterministic local fallback offline, so every
   test runs with no keys and no network).
2. The filters are applied in SQL over ``parts`` join ``verdicts``: type
   tokens match through a small alias table, voltage/current within a
   tolerance, interface tokens through ``interface_variants``.
3. Ranking is VERIFIED first, then NEEDS_MEASUREMENT, then how completely
   a row satisfies the request. ABSTAIN parts (which the gate could not
   prove) never match either, and **HAZARD parts are never returned**.
4. If nothing in the inventory matches, ``common_sources.fallback_matches``
   advises from a static table of household devices instead.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from salvage.inventory import filters as _filters
from salvage.inventory.common_sources import fallback_matches
from salvage.inventory.models import Device, Match
from salvage.schemas import Part, Reading, State, Verdict

DEFAULT_DB_PATH = Path("data/inventory.db")

# Ratings only ever come from the readers' key_rating, where ``2300KV``
# must not read as 2300 V and ``25VA`` must not read as 25 A. The unit
# letters are case-sensitive: upper-case V/A, lower-case m prefix, so a KV
# or VA suffix never glues onto the number.
_RATING = r"(\d+(?:\.\d+)?)\s*"
_VOLTS_RE = re.compile(_RATING + r"(mV|V)\b")
_AMPS_RE = re.compile(_RATING + r"(mA|A)\b")

_VOLT_TOLERANCE_FLOOR = 1.0  # volts: an 11.1 V pack should answer 12 V
_AMPS_TOLERANCE_FLOOR = 0.5  # amps
_AMPS_TOLERANCE_SHARE = 0.2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id     TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    source TEXT,
    notes  TEXT
);

CREATE TABLE IF NOT EXISTS parts (
    id          TEXT PRIMARY KEY,
    device_id   TEXT NOT NULL REFERENCES devices(id),
    part_type   TEXT NOT NULL DEFAULT '',
    label       TEXT NOT NULL DEFAULT '',
    voltage     REAL,
    current     REAL,
    interface   TEXT NOT NULL DEFAULT '',
    crop_path   TEXT,
    search_text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_parts_device ON parts(device_id);

CREATE TABLE IF NOT EXISTS verdicts (
    part_id               TEXT PRIMARY KEY REFERENCES parts(id),
    state                 TEXT NOT NULL,
    label                 TEXT NOT NULL,
    reasons               TEXT NOT NULL DEFAULT '[]',
    requested_measurement TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_state ON verdicts(state);

CREATE TABLE IF NOT EXISTS readings (
    part_id    TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    part_type  TEXT,
    key_rating TEXT,
    condition  TEXT,
    evidence   TEXT,
    PRIMARY KEY (part_id, model_id)
);
"""

_STATE_RANK_SQL = "CASE v.state WHEN 'VERIFIED' THEN 0 ELSE 1 END"


@dataclass
class _Row:
    """One fetched inventory row plus its request-specific score."""

    row: sqlite3.Row
    score: int


def rating_volts(text: str) -> float | None:
    """First volts rating in a key_rating string; ``mV`` scales down.

    ``"2300KV"`` returns None: the KV suffix never reads as volts.
    """
    match = _VOLTS_RE.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2) == "mV" else value


def rating_amps(text: str) -> float | None:
    """First current rating in a key_rating string; ``mA`` scales down."""
    match = _AMPS_RE.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2) == "mA" else value


def interfaces_in(text: str) -> list[str]:
    """Ordered interface tokens that appear in a blob of reading text."""
    return [
        token
        for token, pattern in _filters.INTERFACE_PATTERNS.items()
        if pattern.search(text)
    ]


def _words(text: str) -> list[str]:
    """Lowercase words of a text blob (the SQL search-text tokenizer)."""
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


class Inventory:
    """SQLite-backed inventory of scanned devices, parts and verdicts."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        if path != Path(":memory:"):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.DatabaseError:
            self._conn.close()
            raise

    # -- in operations -------------------------------------------------------

    def add(
        self,
        device: Device,
        parts: Sequence[Part],
        verdicts: Sequence[Verdict],
        *,
        readings: Sequence[Reading] | None = None,
    ) -> None:
        """Remember one scanned device with its parts, verdicts and readings.

        A part without a verdict is stored (for the record) but gets no
        state, so it can never surface in a query. Optional readings feed
        the derived columns: part_type by majority vote, ratings parsed
        from ``key_rating`` only, interface tokens from anything seen.
        """
        with self._conn as conn:
            conn.execute(
                "INSERT OR REPLACE INTO devices (id, name, source, notes) "
                "VALUES (?, ?, ?, ?)",
                (device.id, device.name, device.source, device.notes),
            )
            verdict_by_part = {verdict.part_id: verdict for verdict in verdicts}
            readings_by_part: dict[str, list[Reading]] = {}
            for reading in readings or ():
                readings_by_part.setdefault(reading.part_id, []).append(reading)

            for part in parts:
                verdict = verdict_by_part.get(part.id)
                part_readings = readings_by_part.get(part.id, [])
                fields = self._part_fields(part, verdict, part_readings)
                conn.execute(
                    "INSERT OR REPLACE INTO parts "
                    "(id, device_id, part_type, label, voltage, current, "
                    "interface, crop_path, search_text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    fields,
                )
                if verdict is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO verdicts "
                        "(part_id, state, label, reasons, requested_measurement) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            verdict.part_id,
                            verdict.state.value,
                            verdict.label,
                            json.dumps(list(verdict.reasons)),
                            verdict.requested_measurement,
                        ),
                    )
                if part_readings:
                    conn.executemany(
                        "INSERT OR REPLACE INTO readings "
                        "(part_id, model_id, part_type, key_rating, condition, "
                        "evidence) VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                reading.part_id,
                                reading.model_id,
                                reading.part_type,
                                reading.key_rating,
                                reading.condition.value,
                                reading.evidence,
                            )
                            for reading in part_readings
                        ],
                    )

    @staticmethod
    def _part_fields(
        part: Part,
        verdict: Verdict | None,
        part_readings: list[Reading],
    ) -> tuple:
        """Derive the parts-table row for one part from its provenance."""
        part_type = _majority_part_type(part_readings)
        label = verdict.label if verdict is not None else part.id

        # Ratings: the largest well-formed numbers across the readings.
        # Only key_rating is consulted; a rail mention ("across the 12 V
        # rail") is context, not the part's own rating.
        volts = [
            rating_volts(reading.key_rating)
            for reading in part_readings
            if reading.key_rating
        ]
        amps = [
            rating_amps(reading.key_rating)
            for reading in part_readings
            if reading.key_rating
        ]
        voltage = max((v for v in volts if v is not None), default=None)
        current = max((a for a in amps if a is not None), default=None)

        # Interfaces: tokens seen anywhere this part was described.
        all_text = " \n".join(
            [part_type, label]
            + [reading.key_rating or "" for reading in part_readings]
            + [reading.evidence or "" for reading in part_readings]
        )
        interface = " ".join(interfaces_in(all_text))

        search_text = " ".join(
            _words(
                " \n".join(
                    [part_type, label, interface]
                    + [reading.part_type or "" for reading in part_readings]
                    + [reading.key_rating or "" for reading in part_readings]
                    + [reading.evidence or "" for reading in part_readings]
                )
            )
        )
        return (
            part.id,
            part.device_id,
            part_type,
            label,
            voltage,
            current,
            interface,
            part.crop_path,
            search_text,
        )

    # -- out operations ------------------------------------------------------

    def query(
        self,
        text: str,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> list[Match]:
        """Answer ``"I need a 12 V motor with a driver"`` with stored Matches.

        VERIFIED first, then NEEDS_MEASUREMENT; HAZARD and ABSTAIN parts
        are never returned. When nothing in the inventory matches, the
        static ``common_sources`` household table answers instead.
        """
        parsed = _filters.parse_filters(text, cache_dir=cache_dir)
        if not parsed.type_tokens and not parsed.interface_tokens:
            return []  # nothing usable was asked; offer no advice
        matches = self._query_inventory(parsed)
        if matches:
            return matches
        return fallback_matches(parsed)

    def _query_inventory(self, parsed: _filters.Filters) -> list[Match]:
        """Match the structured filters in SQL; rank in Python."""
        conditions: list[str] = []
        params: dict[str, object] = {}

        # The window: only states the gate could defend.
        conditions.append("v.state IN ('VERIFIED', 'NEEDS_MEASUREMENT')")

        for i, token in enumerate(parsed.type_tokens):
            variants = _filters.token_variants(token)
            group = " OR ".join(
                f"(p.search_text LIKE :type_{i}_{j}"
                f" OR p.part_type LIKE :type_{i}_{j})"
                for j in range(len(variants))
            )
            conditions.append(f"({group})")
            for j, variant in enumerate(variants):
                params[f"type_{i}_{j}"] = f"%{variant}%"

        if parsed.voltage is not None:
            tolerance = max(_VOLT_TOLERANCE_FLOOR, 0.1 * parsed.voltage)
            conditions.append(
                "(p.voltage IS NOT NULL AND ABS(p.voltage - :voltage) <= :volt_tol)"
            )
            params["voltage"] = parsed.voltage
            params["volt_tol"] = tolerance

        if parsed.current is not None:
            tolerance = max(_AMPS_TOLERANCE_FLOOR, _AMPS_TOLERANCE_SHARE * parsed.current)
            conditions.append(
                "(p.current IS NOT NULL AND ABS(p.current - :current) <= :amp_tol)"
            )
            params["current"] = parsed.current
            params["amp_tol"] = tolerance

        for i, token in enumerate(parsed.interface_tokens):
            variants = _filters.interface_variants(token)
            group = " OR ".join(
                f"p.interface LIKE :iface_{i}_{j}" for j in range(len(variants))
            )
            conditions.append(f"({group})")
            for j, variant in enumerate(variants):
                params[f"iface_{i}_{j}"] = f"%{variant}%"

        sql = f"""
            SELECT p.*, v.state, v.label AS verdict_label,
                   v.reasons, v.requested_measurement,
                   d.name AS device_name
            FROM parts p
            JOIN verdicts v ON v.part_id = p.id
            JOIN devices d ON d.id = p.device_id
            WHERE {' AND '.join(conditions)}
            ORDER BY {_STATE_RANK_SQL}, p.id
        """
        scored = [
            _Row(row=row, score=self._row_score(row, parsed))
            for row in self._conn.execute(sql, params).fetchall()
        ]
        scored.sort(
            key=lambda item: (
                0 if item.row["state"] == "VERIFIED" else 1,
                -item.score,
                item.row["id"],
            )
        )
        return [self._row_to_match(item.row) for item in scored]

    @staticmethod
    def _row_score(row: sqlite3.Row, parsed: _filters.Filters) -> int:
        """How completely one row satisfies the request (tie-break only;
        the SQL already guarantees every listed token matched somewhere)."""
        search_text: str = row["search_text"]
        interface: str = row["interface"] or ""
        type_hits = sum(
            1
            for token in parsed.type_tokens
            if any(
                variant in search_text
                or variant in (row["part_type"] or "")
                for variant in _filters.token_variants(token)
            )
        )
        iface_hits = sum(
            1
            for token in parsed.interface_tokens
            if any(variant in interface for variant in _filters.interface_variants(token))
        )
        # Slight preference for exact state solidity on equal coverage:
        return type_hits + iface_hits

    @staticmethod
    def _row_to_match(row: sqlite3.Row) -> Match:
        reasons = json.loads(row["reasons"]) if row["reasons"] else []
        return Match(
            source="inventory",
            part_id=row["id"],
            part_type=row["part_type"] or None,
            label=row["verdict_label"],
            state=State(row["state"]),
            device_id=row["device_id"],
            device=row["device_name"],
            voltage=row["voltage"],
            current=row["current"],
            interface=row["interface"] or None,
            crop_path=row["crop_path"],
            requested_measurement=row["requested_measurement"],
            reasons=reasons,
        )

    def close(self) -> None:
        self._conn.close()


def _majority_part_type(part_readings: list[Reading]) -> str:
    """The readers' most-agreed part_type; empty when no one named it."""
    counts: dict[str, int] = {}
    for reading in part_readings:
        if reading.part_type:
            counts[reading.part_type] = counts.get(reading.part_type, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
