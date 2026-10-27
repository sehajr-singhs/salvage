"""Static fallback answers (brief 05): household devices that contain a part.

When the inventory has no part that matches a request, ``fallback_matches``
answers from this static table instead: which household devices usually
contain the requested kind of part, best first. It is advice, not verified
stock — those Matches have no part_id and no state.

Deliberately mains-safe: no CRTs, microwaves, or mains-primary power
supplies appear here, because the gate refuses those teardowns and this
table must never point a person at one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_from salvage.filters import (
    Filters,  # noqa: F401
)
