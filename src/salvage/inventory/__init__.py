"""The Salvage inventory: remember every scanned part, answer reverse searches.

Public surface (brief 05):

- ``Inventory(db_path=None)`` — SQLite store at ``data/inventory.db`` by
  default (pass ``db_path`` in tests). ``.add(device, parts, verdicts,
  readings=None)`` remembers one scanned device; ``.query(text,
  cache_dir=None) -> list[Match]`` answers requests like "I need a 12 V
  motor with a driver".
- Module-level ``add(...)`` / ``query(...)`` conveniences over the default
  store, so callers that do not manage a store still have the brief's
  literal interface.
- ``parse_filters`` / ``Filters`` — the one cached LLM call that turns a
  free-text request into structured filters (type, voltage, current,
  interface), with a deterministic no-key heuristic fallback.
- ``COMMON_SOURCES`` / ``fallback_matches`` — when nothing in the inventory
  matches, static advice about household devices that usually contain the
  requested part kind.

Never returned: HAZARD parts (and ABSTAIN parts, which the gate has
unproven). Matches rank VERIFIED first, then NEEDS_MEASUREMENT; every
inventory Match names the device to open and the part_id to pull.
"""

from salvage.inventory import filters, store
from salvage.inventory.common_sources import (
    COMMON_SOURCES,
    CommonSource,
    fallback_matches,
)
from salvage.inventory.filters import Filters, parse_filters
from salvage.inventory.models import Device, Match
from salvage.inventory.store import DEFAULT_DB_PATH, Inventory

_DEFAULT: Inventory | None = None


def _default_inventory() -> Inventory:
    """One lazily-created store at the default path for module-level calls."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Inventory()
    return _DEFAULT


def add(device, parts, verdicts, *, readings=None) -> None:
    """``Inventory.add`` against the default ``data/inventory.db`` store."""
    _default_inventory().add(device, parts, verdicts, readings=readings)


def query(text: str, *, cache_dir=None) -> list[Match]:
    """``Inventory.query`` against the default ``data/inventory.db`` store."""
    return _default_inventory().query(text, cache_dir=cache_dir)


__all__ = [
    "COMMON_SOURCES",
    "DEFAULT_DB_PATH",
    "CommonSource",
    "Device",
    "Filters",
    "Inventory",
    "Match",
    "add",
    "fallback_matches",
    "filters",
    "parse_filters",
    "query",
    "store",
]
