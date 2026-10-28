"""The bench measurement dialogue (brief 08): ask, listen, parse, re-grade.

When the gate leaves a part NEEDS_MEASUREMENT, the agent asks one cheap
question out loud — ``"Put the meter across the two motor leads. I expect
0.5 to 20 ohms."`` — and waits for the number (text input first, voice
later: the :class:`Prompter` seam is where a Muse Voice Transcribe client
plugs in, without any other change).

Three pieces live here, deliberately model-free:

- :func:`parse_measurement` turns one free-text reply into a number with
  its unit (``"8.2 ohms"``, ``"3.7 V"``, ``"spins"``, ``"beeps"``,
  ``"open circuit"``); ``None`` when nothing measurable was said, and the
  dialogue asks once more before giving up on that part.
- :func:`asks` decides *who* gets asked what: every NEEDS_MEASUREMENT
  verdict (the gate's own instruction), plus an ABSTAIN verdict whose
  consensus type has a cheap check in ``salvage.gate.checks`` — that is a
  part the gate had to refuse because one reading failed or contradicted
  the rest, and it stays resolvable by the same meter. HAZARD parts are
  never asked, fully.
- :func:`measure` runs the loop: ask, parse, retry once, skip politely.
  It returns replies keyed by part id, already normalized to the phrases
  :func:`salvage.gate.decide.re-run` parses (the canonical phrases are the
  ones the gate's own parsers accept, so the two modules never
  disagree).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from salvage.gate.checks import find_check
from salvage.schemas import Reading, State, Verdict

# The phrase handed back when a meter says nothing readable. Every parser
# in salvage.gate.checks reads this as 0.0 (open winding, dead cell, blown
# fuse), which is exactly what a broken part looks like — cautious, never
# confident.
OPEN_CIRCUIT_PHRASE = "open circuit"

_NUMBER = r"(\d+(?:\.\d+)?)"
_NO_SPIN = re.compile(
    r"\b(no|not|doesn'?t|doesn't|did not|didn'?t|never|won'?t|stuck|seiz\w+|"
    r"jamm\w+|stall\w+|lock\w*)\b",
    re.IGNORECASE,
)
_SPIN = re.compile(r"\bspins?\b|\bspun\b|\brotat\w+", re.IGNORECASE)
_CONTINUITY = re.compile(r"\bbeeps?\b|\bbuzz\w+|\bcontinuit\w+", re.IGNORECASE)
_NO_CONTINUITY = re.compile(
    r"\bopen\b|\bno\s*continu\w+|\bblown\b|\bpop(?:ped)?\b", re.IGNORECASE
)
_VOLTS = re.compile(_NUMBER + r"\s*(?:volts?\b|v\b)", re.IGNORECASE)
_OPEN_CIRCUIT = re.compile(
    r"\bopen(?:[ -]?circuit)?\b|\bo\.?l\.?\b|\bno\s*continu\w+\b", re.IGNORECASE
)
_OHMS = re.compile(
    _NUMBER + r"\s*(k|kilo|mega|m)?\s*ohms?\b", re.IGNORECASE
)
_OHMS_SYMBOL = re.compile(_NUMBER + r"\s*Ω")
_BARE_NUMBER = re.compile(_NUMBER + r"\s*(k|mega|m)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedMeasurement:
    """One measurable reply: the value, its unit, and how to say it back."""

    value: float
    """The number the user measured (0.0 for an open circuit / 'no')."""

    unit: str
    """``"ohm"`` | ``"volt"`` | ``"spin"`` | ``"continuity"``."""

    phrase: str
    """The canonical free-text phrase handed to ``gate.decide``."""


def _ohms(value: float) -> ParsedMeasurement:
    return ParsedMeasurement(value=value, unit="ohm", phrase=f"{value:g} ohms")


def _volts(value: float) -> ParsedMeasurement:
    return ParsedMeasurement(value=value, unit="volt", phrase=f"{value:g} V")


def parse_measurement(text: str) -> ParsedMeasurement | None:
    """Parse one free-text reply into a number with its unit.

    Keyword phrases win over numbers (``"spins"`` is a spin test result
    whatever the scale reading was, ``"open circuit"`` is 0). An unprefixed
    number defaults to ohms — the most common bench check — and the gate's
    own per-type parser still reads the phrase its own way, so a wrong
    guess here can only ever move the gate *toward* caution, never toward
    a false verify. Returns ``None`` when nothing measurable was said.
    """
    t = (text or "").strip()
    if not t:
        return None

    # 1. spin test: the word "spin" carries the whole result
    if _SPIN.search(t):
        negative = bool(_NO_SPIN.search(t))
        return ParsedMeasurement(
            value=0.0 if negative else 1.0,
            unit="spin",
            phrase="the fan does not spin" if negative else "the fan spins freely",
        )

    # 2. continuity: a beep is a result, not a number
    if _CONTINUITY.search(t):
        negative = bool(_NO_CONTINUITY.search(t))
        return ParsedMeasurement(
            value=0.0 if negative else 1.0,
            unit="continuity",
            phrase=OPEN_CIRCUIT_PHRASE if negative else "the meter beeps: continuity",
        )

    # 3. an explicit volt reading
    volts = _VOLTS.search(t)
    if volts is not None:
        return _volts(float(volts.group(1)))

    # 4. an open circuit: 0 in every unit, always cautious about it
    if _OPEN_CIRCUIT.search(t):
        return ParsedMeasurement(
            value=0.0, unit="ohm", phrase=OPEN_CIRCUIT_PHRASE
        )

    # 5. an explicit ohm reading, with the k/m multiplier when written
    match = _OHMS.search(t) or _OHMS_SYMBOL.search(t)
    if match is not None:
        value = float(match.group(1))
        prefix = (match.group(2) or "").lower()
        if prefix in ("k", "kilo"):
            value *= 1000.0
        elif prefix in ("m", "mega"):
            value *= 1_000_000.0
        return _ohms(value)

    # 6. a bare number resolves to ohms (see the docstring)
    bare = _BARE_NUMBER.search(t.strip())
    if bare is not None:
        value = float(bare.group(1))
        prefix = (bare.group(2) or "").lower()
        if prefix in ("k", "kilo"):
            value *= 1000.0
        elif prefix in ("m", "mega"):
            value *= 1_000_000.0
        return _ohms(value)

    return None


# --- prompters -------------------------------------------------------------

class Prompter(Protocol):
    """How the dialogue reaches the user (text now, voice later)."""

    def ask(self, part_id: str, question: str) -> str | None:
        """Ask one question; return the reply, or ``None`` to skip the part."""
        ...


class ConsolePrompter:
    """Text terminal input with a plain question line.

    EOF / a closed terminal returns ``None`` (skip the part): the dialogue
    never crashes on an absent user, per the 00-README failure rule.
    """

    def ask(self, part_id: str, question: str) -> str | None:
        try:
            print(f"{part_id} > {question}")
            return input("measurement: ")
        except (EOFError, KeyboardInterrupt, OSError):
            print()  # keep the next block on its own line
            return None


class ScriptPrompter:
    """A reply script keyed by part id, for tests and offline demos.

    The value is either one string (returned every time) or a list of
    strings consumed in order (so a test can script "unparseable,
    then the real measurement"). Missing parts return ``None``.
    """

    def __init__(self, replies: Mapping[str, str | list[str]]) -> None:
        supplied: dict[str, list[str]] = defaultdict(list)
        for part_id, reply in replies.items():
            if isinstance(reply, list):
                supplied[part_id] = list(reply)
            else:
                supplied[part_id] = [reply]
        self._replies = supplied
        self.asked: list[tuple[str, str]] = []

    def ask(self, part_id: str, question: str) -> str | None:
        self.asked.append((part_id, question))
        queue = self._replies.get(part_id)
        if not queue:
            return None
        reply = queue.pop(0)
        return reply or None


# --- who gets asked what ----------------------------------------------------

_GENERIC_ASK = (
    "Report one direct observation that proves this part works (its full "
    "marking and any rating, and what you probed)."
)


@dataclass(frozen=True)
class Ask:
    """One question the dialogue wants to ask the user about one part."""

    part_id: str
    state: State
    question: str


def consensus_type(part_id: str, readings: Sequence[Reading]) -> str:
    """The readers' most-agreed part_type for one part (empty if unnamed)."""
    counts: Counter[str] = Counter(
        reading.part_type for reading in readings if reading.part_id == part_id
    )
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def asks(
    verdicts: Sequence[Verdict],
    readings: Sequence[Reading],
) -> list[Ask]:
    """The questions worth asking this user, in verdict order.

    NEEDS_MEASUREMENT verdicts are asked with the gate's own instruction.
    ABSTAIN verdicts are asked only when they name the measurement they
    still want, or when the readers' consensus type has a cheap check in
    ``gate.checks`` — those are resolvable by the same meter. HAZARD parts
    are never asked: there is no measurement that makes a swollen cell
    safe, and the question would be the wrong instruction.
    """
    result: list[Ask] = []
    for verdict in verdicts:
        if verdict.state is State.HAZARD:
            continue
        if verdict.state is State.NEEDS_MEASUREMENT:
            result.append(
                Ask(
                    part_id=verdict.part_id,
                    state=verdict.state,
                    question=verdict.requested_measurement or _GENERIC_ASK,
                )
            )
            continue
        if verdict.state is not State.ABSTAIN:
            continue  # VERIFIED: nothing to ask; the rest is refused/asked above
        question = verdict.requested_measurement
        if question is None:
            ptype = consensus_type(verdict.part_id, readings)
            check = find_check(ptype) if ptype else None
            if check is None:
                continue  # nothing the same meter could settle either way
            question = check.instruction
        if question:
            result.append(
                Ask(part_id=verdict.part_id, state=verdict.state, question=question)
            )
    return result


def measure(
    verdicts: Sequence[Verdict],
    readings: Sequence[Reading],
    prompter: Prompter | None = None,
    *,
    retries: int = 1,
) -> dict[str, str]:
    """Run the dialogue; return ``{part_id: canonical measurement phrase}``.

    An unparseable reply is asked once more (``retries`` replies in total);
    a part the user skips or whose replies stay unparseable is simply left
    out of the dict, which leaves its verdict exactly as the gate wrote
    it. A skipped part never becomes more wrong than it was.
    """
    prompter = prompter if prompter is not None else ConsolePrompter()
    answers: dict[str, str] = {}
    for ask in asks(verdicts, readings):
        for _ in range(retries + 1):
            reply = prompter.ask(ask.part_id, ask.question)
            if reply is None:
                break  # the user skipped this part
            parsed = parse_measurement(reply)
            if parsed is not None:
                answers[ask.part_id] = parsed.phrase
                break
    return answers


# --- before/after ----------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    """One part's state before and after the dialogue re-ran the gate."""

    part_id: str
    before: State
    after: State


def state_transitions(
    before: Sequence[Verdict],
    after: Sequence[Verdict],
) -> list[Transition]:
    """Every part whose state actually moved between two gate runs."""
    before_state = {verdict.part_id: verdict.state for verdict in before}
    return [
        Transition(
            part_id=verdict.part_id,
            before=before_state[verdict.part_id],
            after=verdict.state,
        )
        for verdict in after
        if verdict.part_id in before_state
        and before_state[verdict.part_id] is not verdict.state
    ]
