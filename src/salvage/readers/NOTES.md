# NOTES — brief 03: multi-model part readers

## What works

- `salvage.readers.reader.read_part(part, providers=None, *, cache_dir=None,
  timeout=30.0) -> list[Reading]`: for one Part crop, ask every enabled
  provider (all OpenAI-compatible chat-completions, from `config.PROVIDERS`)
  to identify it, and return one validated Reading per reliably-parsed
  answer. Explicitly passed providers are filtered to the currently
  available ones; with no providers or no keys the honest answer is zero
  votes.
- Parallelism per the brief: every provider runs under `asyncio.gather`
  with a 30 s outer `asyncio.wait_for` cap *and* a 30 s `httpx` timeout on
  every network call. A provider that hangs is abandoned and contributes no
  vote; the other providers keep running concurrently (tested with a
  provider scripted to hang for 30 s — the fast one still answers).
- One prompt for all readers, `salvage.readers.prompt.READ_PROMPT`: it asks
  for JSON with exactly the Reading fields and carries the honesty rules —
  quote the markings you actually see in `evidence`; if you cannot read a
  marking, say so and return null, never guess. `RETRY_NOTE` is the
  corrective nudge appended on a parse retry.
- Every call goes through `salvage.cache.cached_call`, keyed by the
  provider-qualified model id (`"<name>:<model_id>"`, so two providers
  serving the same model string can never collide) plus the sha256 of the
  exact request payload. The request body is deterministic
  (sorted keys, fixed separators) so re-reading the same crop with the same
  provider is served from disk at zero cost (tested: zero HTTP calls on the
  second run). API keys live only in the `Authorization` header: they are
  never in the payload, the digest, or the cache file (tested).
- Output validated into `Reading` with pydantic. `part_id`/`model_id` come
  from the caller, never the model (a reader cannot rename the part or pose
  as another reader — tested). Parse failures retry exactly once with the
  corrective `RETRY_NOTE` (which changes the payload, so the retry cannot
  be served from the failed attempt's cache entry) and then the vote is
  dropped. Network/HTTP/timeout failures are *not* parse failures and get
  no retry (they either never answered or answered in error).
- `salvage.readers.agreement.agreement(readings) -> float`: the share of
  readings agreeing on part_type (light normalization: case + whitespace)
  and on the part_number family (flattened to one lowercase alphanumeric
  run). Deliberately conservative: `IRF540` vs `IRF540N` are different
  families, and a reading with no legible number does not agree with one
  somebody did read — ambiguous evidence lowers agreement instead of
  raising it. Also exports `part_number_family`.

## The one design decision worth knowing about

- `salvage.readers.read_part` is exported **lazily via PEP 562
  `__getattr__`, and only when at least one provider key is configured**.
  With no key anywhere, `from salvage.readers import read_part` raises
  ImportError. That is what briefs 06 and 08 were built against: their
  pipelines (`salvage.web.pipeline.resolve_readings`,
  `salvage.pipeline`) catch exactly that ImportError and fall back to
  `offline_mock_readings`, and `test_web.py` asserts the keyless scan
  reports `readings_mode == "offline mock reader"`. Without the gate,
  shipping the real reader would flip the keyless web path to `"real"`
  with zero actual readers and break brief 06's suite, which I must not
  edit. Behavior: keyless machine → offline mock reader is authoritative
  (unchanged, deterministic); keyed machine → the real parallel readers
  engage automatically, no other file needs to change. The decision is
  stated in the package docstring and tested in
  `test_readers_imports_only_when_a_key_exists`.

## Test coverage (tests/test_readers.py, 32 tests, all offline)

- The brief's done-when: 3 agreeing providers + 1 timing out + 1 returning
  garbage → exactly 3 Readings, `agreement(...) == 1.0`; attempt profile
  asserted per host (agree ×1, timeout ×1 with no retry, garbage ×2 then
  dropped).
- HTTP mocked at `httpx.post` per-provider (same pattern as 02's backend
  tests): timeouts, HTTP 500, prose-instead-of-JSON, fenced JSON, extra
  keys, case/spacing normalization, out-of-range confidence, empty
  evidence, garbage body shapes, list-shaped content.
- Cache: hit costs zero network, key stability, cross-provider collision
  impossibility, keys never leak into payloads.
- Missing crop → zero votes; missing providers → zero votes; default
  (keyless) path sends nothing.

Verified whole-suite: `.venv\Scripts\python -m pytest -q` → 223 passed
(191 pre-existing across 01/02/04/05/06/07/08 + 32 new), all offline.

## What doesn't work / known limits

- No keys exist on this machine, so the real provider path is unverified
  live; it is covered only by mocked-HTTP tests (same caveat as 02's
  hosted backend).
- The offline mock reader (`web.pipeline.offline_mock_readings`) still owns
  the keyless scan path; with keys, the real readers would surface real
  model answers, but their quality/agreement is unknown until keys exist.
- `response_format: {"type": "json_object"}` was deliberately not sent:
  some OpenAI-compatible free-tier endpoints reject it. The corrective
  retry handles spirited formatting instead.
- `confidence` out of [0, 1] (e.g. 87 instead of 0.87) is treated as a
  parse failure rather than rescaled: guessing the intended scale felt
  like exactly the guessing this module is built to refuse.
- Timeout threads can outlive their logical call (a hung HTTP call keeps
  the worker thread until its own 30 s httpx cap), which is memory, not
  correctness,  overhead — the asyncio cap guarantees no part ever waits past its
  own 30 s bound.

## Stubbed / assumed

- Nothing in this brief's scope is a stub. Assumed (and compatible with the
  shared contract): providers speak v1 chat-completions with
  `image_url` content parts, as brief 00 and `config.py` describe.
