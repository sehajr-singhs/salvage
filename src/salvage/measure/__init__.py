"""The bench measurement dialogue (brief 08).

Public surface:

- ``parse_measurement(text) -> ParsedMeasurement | None`` — one free-text
  meter reply (``"8.2 ohms"``, ``"3.7 V"``, ``"spins"``, ``"open circuit"``)
  into a value with its unit, or ``None`` when nothing measurable was said.
- ``Prompter`` / ``ConsolePrompter`` / ``ScriptPrompter`` — how the agent
  reaches the user: terminal text input today, a Muse Voice Transcribe
  client behind the same seam later.
- ``asks(...)`` / ``measure(...)`` — the ask loop over a verdict list
  (NEEDS_MEASUREMENT always, ABSTAIN when a cheap check could still settle
  it, HAZARD never), with one retry and polite skips.
- ``Ask`` / ``Ask(part_id, state, question)``, ``ParsedMeasurement``,
  ``Transition`` / ``state_transitions(...)` — the small data types the
  dialogue and the pipeline exchange.
"""

from salvage.measure.dialogue import (
    Ask,
    ConsolePrompter,
    OPEN_CIRCUIT_PHRASE,
    ParsedMeasurement,
    Prompter,
    ScriptPrompter,
    State,
    Verdict,
    Transition,
    consensus_type,
    measure,
    parse_measurement,
    state_transitions,
)

__all__ = [
    "OPEN_CIRCUIT_PHRASE",
    "Ask",
    "ConsolePrompter",
    "ParsedMeasurement",
    "Prompter",
    "ScriptPrompter",
    "State",
    "Transition",
    "Verdict",
    "consensus_type",
    "measure",
    "parse_measurement",
    "state_transitions",
]
