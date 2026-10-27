# NOTES.md — 04: verify-or-abstain gate

## What works

- `decide(part, readings, measurement=None) -> Verdict` in `src/salvage/gate/decide.py`,
  deterministic, no models, no network. It applies the brief's rules strictly
  in order:
  1. **HAZARD** via `gate/hazards.py`: a table of four rules (swollen/damaged
     lithium cell — guarded so the word "swollen" only fires with cell context;
     CRT/flyback; microwave HV capacitor/magnetron; mains primary side), each
     matched against the claimed part_type and evidence text from every
     reading, plus the measurement note. The reason always includes the rule's
     "instead" action, so the verdict says what to do, not just that the gate
     refuses. Hazard wins even with a single reading and even over quorum
     failures.
  2. **ABSTAIN** when fewer than 2 valid readings vote or agreement on the
     plurality part_type is below 0.67 (so 2-of-3 = 0.67 fails, 3-of-3 or
     2-of-2 passes). Also abstains when agreeing readers contradict each other
     on condition (works vs dead).
  3. **NEEDS_MEASUREMENT** when the readers agree but would report an unknown
     condition, or the type has a cheap check in `gate/checks.py` (motor
     winding 0.5–20 ohms; Li-ion cell OCV 2.5–4.25 V; fan spins at rated
     voltage; fuse continuity; capacitor ESR under 5 ohms) and no usable
     measurement has arrived. `requested_measurement` always carries the
     plain-English instruction including the expected range.
  4. **VERIFIED** only when the readers agree and either the parsed
     measurement is in range or the type needs no measurement.An in-range measurement also promotes an "unknown" reading set to VERIFIED (that is
     the D3 behavior: one cheap measurement settles the abstention). An
     out-of-range measurement contradicts a "works" claim (ABSTAIN) and
     confirms a "dead" claim (VERIFIED, expected-range violation consistent
     with dead, e.g. an open motor winding).
  5. **Consistency downgrades** in `gate/checks.py` (`consistency_problems`):
     a capacitor rated below the rail a reader says it sits on (only when
     both voltages are actually present in the evidence / key_rating), or a
     part_number matching none of its type family's marking patterns. Both
     downgrade to ABSTAIN, keeping the measurement ask so the bench agent can
     still propose it.
- Missing evidence can only make the gate more cautious: one vote per reader
  model (duplicate model_ids collapse and never inflate agreement), readings
  for other parts never vote, fewer than 2 votes abstains, unparseable
  measurements are treated the same as no measurement, and a missing
  part_number never triggers the family downgrade.
- Tests: `tests/test_gate.py` — 38 table-driven scenarios covering all four
  states and every downgrade/caution rule, plus invariant tests
  (determinism, hazard-beats-quorum, unparseable-m == no-m) and unit tests
  for the parser and table modules. 66 tests here; 94 across the whole repo.
  All pass with `python -m pytest`, fully offline.

## What doesn't work / known limits

- The measurement is free text parsed with regexes (`parse_ohms`,
  `parse_volts`, spin/continuity keyword rules). A chatty sentence like
  "measured on the 200 ohm scale, 8.2" would parse the "200" first and
  abstain; the failing path is cautious but not clever. The 05-measure brief
  should pass structured values (`{"value": 8.2, "unit": "ohm"}` serialized)
  through `decide()` for real precision — the current string interface is the
  stub boundary.
- Family patterns are deliberately coarse (e.g. any 4-digit run counts as a
  motor/fan marking). They are a mismatch detector, not a datasheet parser.
- The rail check only sees rails that readers explicitly mention in evidence;
  there is no board model yet (that would belong to the foreman's plan).
- The "dead" consensus without a check is VERIFIED dead, since the readers
  unanimously agree and there is no cheap opposite-check; the reviewers flag
  it in the reason and the label carries "(dead)".

## What's stubbed

- Nothing in the 04 scope. Every table row exists and is tested; the only
  future-facing gap is structured measurement parsing, noted above.
