# Campaign 2 Phase 05 — Tools-server packaging

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The tools server installs from its own `pyproject.toml` and shares the root uv
workspace lockfile. Dependencies have one declaration rather than a second
manually maintained Dockerfile list. The flat module layout remains importable.

Docker separates dependency and source installation so ordinary source edits
reuse cached dependencies. Package source copying must include new modules;
otherwise a passing laptop import can still become a container startup failure.
