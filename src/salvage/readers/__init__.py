"""Brief 03: the multi-model part readers.

Public surface: the honest per-part agreement signal
(:func:`salvage.readers.agreement.agreement`) and, conditionally,
:func:`salvage.readers.reader.read_part` — the parallel, cached reader.

``read_part`` is exported lazily (PEP 562 ``__getattr__``) and only when
at least one provider key is configured in the environment. With no key
anywhere it raises ImportError, which is what briefs 06 and 08 build
against: their pipelines catch that ImportError and fall back to the
deterministic offline mock reader, so a keyless machine keeps the exact
offline behavior those briefs specify (their no-key web test asserts the
``offline mock reader`` mode) while a keyed box flips to the real
readers. With no key nothing here pretends to have looked at a part:
the mock is the honest answer, not a stand-in that guesses.
"""

from __future__ import annotations

from salvage.readers.agreement import agreement, part_number_family
from salvage.readers.prompt import READ_PROMPT, RETRY_NOTE

__all__ = [
    "READ_PROMPT",
    "RETRY_NOTE",
    "agreement",
    "part_number_family",
    "read_part",
]


def __getattr__(name: str):  # noqa: ANN201 - PEP 562 hook
    if name == "read_part":
        from salvage.config import PROVIDERS

        if not any(p.is_available() for p in PROVIDERS):
            # No provider key is configured: nothing here has actually
            # looked at any crop, so the surface stays closed and the
            # callers' keyless fallback (the offline mock reader) stays
            # authoritative, exactly as briefs 06 and 08 define it.
            raise ImportError(
                "salvage.readers.read_part needs at least one provider "
                "key in the environment; with no keys the offline mock "
                "reader (salvage.web.pipeline.offline_mock_readings) "
                "answers instead"
            )
        from salvage.readers.reader import read_part

        return read_part
    raise AttributeError(f"module 'salvage.readers' has no attribute {name!r}")
