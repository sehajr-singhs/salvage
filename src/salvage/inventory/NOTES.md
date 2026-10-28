# NOTES — brief 05: inventory and reverse search

## What works

- `Inventory(db_path=None)` in `store.py`: a SQLite store at
  `data/inventory.db` by default (tests pass `:memory:` or a tmp path).
  Tables `devices`, `parts`, `verdicts`, plus a `readings` table for
  provenance (the brief required the first three; readings are stored so a
  re-scan can reconstruct how a rating or part_type was derived).
- `add(device, parts, verdicts, *, readings=None)` remembers one scanned
  device and derives searchable columns from the readings: `part_type` by
  majority vote across readers, `voltage`/`current` parsed only from
  `key_rating` (unit-aware: `2300KV` is never 2300 V, `25VA` never 25 A,
  `mV`/`mA` scale down), and `interface` tokens (i2c, spi, esc, driver, ...)
  from anything the readers saw. A part without a verdict is stored but
  has no state, so a query can never surface it.
- `query(text, cache_dir=None) -> list[Match]`, exactly the pipeline the
  brief names: `parse_filters` turns the request into structured filters
  (type, voltage, current, interface) with one cached LLM call through
  `salvage.cache`, then the filters run in SQL over `parts` join
  `verdicts` join `devices` (type via alias variants, voltage/current
  within a tolerance — an 11.1 V pack answers a 12 V request, a 48 V one
  does not —, interface via `interface_variants`). Ranking is VERIFIED
  first, then NEEDS_MEASUREMENT, then how completely a row satisfies the
  request; HAZARD and ABSTAIN parts are filtered out in SQL and are never
  returned. Each Match names the device to open (`device`, `device_id`)
  and the `part_id` to pull, and carries the gate's reasons and, for
  NEEDS_MEASUREMENT, the `requested_measurement` so the bench can ask
  out loud.
- `common_sources.py`: a static, deliberately mains-safe table of 17
  household device families (toy cars, tray-loading CD drives, old
  laptops, printers, routers, ...) with the part kinds they contain, plus
  `fallback_matches(filters)`, used whenever the inventory has no hit.
  Fallback Matches carry no part_id and no state, flag interface gaps and
  voltage mismatches in their reasons, and never point at mains-side
  teardowns (the table is locked by a test).
- Filters (`filters.py`, already half-committed): one cached LLM call per
  distinct request, payload hashed with the reader model id; the call is
  only attempted when `META_API_KEY` exists in `.env`, and every failure —
  no key, timeout, non-200, unparseable or empty reply — falls back to a
  deterministic local heuristic parser. Two bugs fixed in it this
  session: `500 mA` now normalizes to 0.5 amps (it used to parse as 500),
  and the parser's unit vocabulary stayed consistent with that change.
- Module-level `inventory.add(...)` / `inventory.query(...)` conveniences
  over a lazily-created default store, matching the brief's literal
  interface.
- `tests/test_inventory.py`: 37 offline tests, including the brief's
  done-when — an exact match on a seeded DB, the fallback path, and
  hazard exclusion — plus state ranking, abstain exclusion, the cached
  LLM round-trip, persistence across a reopened DB file, and the mains
  safety lock. Full suite: 152 passed.

## What doesn't (yet)

- No live model call has ever been made: no keys on this machine. The LLM
  filter path is exercised only with a stubbed parser in tests; the real
  prompt/HTTP shape in `_llm_parse` is a documented guess (OpenAI-
  compatible `chat/completions`, strict JSON reply) until a Meta API key
  exists.
- `parse_filters` sees only the text of a request. A request like "the
  12 V thing from the drone yesterday" (temporal/device references) is
  out of scope; the filters vocabulary it returns has no slots for that.
- Inventory rows store only what the readers said at scan time. If a
  part state changes later (a measurement resolving a NEEDS_MEASUREMENT),
  the caller (brief 08 wiring) must re-add the device; nothing here
  watches for updates beyond `INSERT OR REPLACE` upserts.

## Stubbed / assumed

- Matching tolerances are heuristics hardcoded in `store.py`: ±max(1 V,
  10%) for voltage, ±max(0.5 A, 20%) for current. They are stated in the
  Match's derivation, not justified by data.
- Voltage/current parsing covers `V`, `mV`, `A`, `mA` with a KV/VA guard
  but not `W` (power) or `UF` — a "10 W driver" request matches by type
  token only.
- The `COMMON_SOURCES` table is static advice written from general
  household knowledge; it has not been validated against the real 50-
  device teardown corpus yet.
- `search_text` is a lowercased word stream over part_type, label,
  interface and all reading text, so LIKE '%motor%' matching can
  surface a part whose readers mentioned "motor" incidentally; coverage
  scoring then decides ordering. No embedding/semantic fallback.
