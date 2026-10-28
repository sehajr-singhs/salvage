"""The Salvage demo surface (brief 06): a black-and-white web UI.

Public surface:

- ``create_app(**deps) -> FastAPI`` — the thin API layer with injectable
  segment backend, readings provider, inventory and paths (for tests and
  the scene-08 wiring). The module-level ``app`` binds the default paths.
- ``salvage.web.pipeline.scan_device`` — photo → parts → readings → gate,
  importing sessions 02 to 05; readers (03) fall back to an offline mock
  reader while that subpackage is a stub.
- ``salvage.web.samples_build.build()`` — regenerates the cached sample
  bundle under ``web/static/samples/``.

Static assets (page, styles, sample bundle) live in ``web/static/`` at the
repo root, served at ``/static``; scan artifacts under the data root are
served at ``/data``.
"""

from salvage.web.app import create_app

__all__ = ["create_app"]
