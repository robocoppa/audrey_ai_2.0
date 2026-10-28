# Campaign 2 Phase 15 — Inline image input

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

Chat input accepts text and typed multimodal content parts. An actual image
in the latest user turn selects the vision-capable fast path before ordinary
text routing, preserving image data that text-only panel reconstruction would
otherwise discard. Capability checks apply to fallback models too.

This is separate from uploaded-image KB retrieval. Campaign 4 extends native
attachments and Responses input while retaining authenticated ownership.
