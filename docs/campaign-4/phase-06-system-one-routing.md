# Campaign 4 Phase 06 — System One router assessment

**Status:** Complete. Keep `qwen3.5:4b`; no production router change is planned.

## Decision

Measured `tev1:0.8b`, `tev1:latest`, `nimble:latest`, `clef:latest`, and `clef-flash:latest` did not clear the complete Audrey routing gate. Clef matched incumbent quality on production-reached cases but failed cold latency and residency. Clef Flash failed cold-start reliability. Tev/Nimble comparisons did not establish a replacement that met the full gate.

Model measurements and publishable comparisons belong in [the model report](../../evals/MODEL-FACTS.md), not duplicate phase logs.

## Conditions for reopening

The probe-local `scripts/probes/systemone_router_probe.py` compares Ollama's typed `/v1/systemone` choices with Audrey's real incumbent router. It adds no production client or dependency.

A replacement must preserve label quality, costly-reasoning misroutes, escalation behavior, cold/warm latency, coexistence with active GPU workers, and failure fallback on Audrey's hardware. System One `confidence` measures distribution concentration; it cannot be copied into the incumbent's confidence/escalation threshold without labeled calibration.

Strong keywords, cheap short-prompt routing, explicit tool-name handling, image compatibility, forced virtual choices, complexity ordering, and weak-keyword/general fallbacks stay authoritative. Any future backend switch needs a configuration rollback.

English-only evaluation applies. Automatic skill-selection research is separate under Phase 03 and remains disabled.
