"""Per-user KB collections (Phase 13).

Each user who uploads files gets two lazily-created Qdrant collections:
  - `kb_user_text_<sanitized>`   — 768-d cosine, nomic-embed
  - `kb_user_images_<sanitized>` — 512-d cosine, CLIP

Sanitization is `[^a-z0-9]+ -> "_"`, lowercased, stripped of leading/
trailing underscores. So `bart@proton.me` → `bart_proton_me`. The raw
user id is preserved in each point's `user` payload field for display
and for the payload-filter guard on deletes.

Payload shape (both collections):
  {
    "user":       "<raw user id>",            # e.g. bart@proton.me
    "file_id":    "<uuid4>",                  # one per uploaded file
    "source":     "/data/uploads/<sanitized>/<uuid><ext>",
    "filename":   "RA_management.pdf",        # original, for display
    "chunk_idx":  N,
    "kind":       "text" | "image",
    "text":       "..." (text only),
    "caption":    "..." (image only),
    "mime":       "application/pdf",
    "bytes":      1048576,
    "uploaded_at":"2026-04-24T19:00:00Z",
    "mtime":      1776000000.0,
  }

Deletes are always double-filtered on `file_id` AND `user` — see
QdrantKB.delete_by_file_id.
"""

from __future__ import annotations

import re

from audrey.kb.qdrant import IMAGE_DIM, TEXT_DIM, QdrantKB

USER_TEXT_PREFIX = "kb_user_text_"
USER_IMAGE_PREFIX = "kb_user_images_"

_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def sanitize_user(user: str) -> str:
    """Lowercase, replace non-alphanumeric runs with `_`, strip edges."""
    s = _SANITIZE_RE.sub("_", user.lower()).strip("_")
    if not s:
        raise ValueError(f"user id sanitizes to empty string: {user!r}")
    return s


def user_text_collection(user: str) -> str:
    return f"{USER_TEXT_PREFIX}{sanitize_user(user)}"


def user_image_collection(user: str) -> str:
    return f"{USER_IMAGE_PREFIX}{sanitize_user(user)}"


async def ensure_user_collections(qdrant: QdrantKB, user: str) -> tuple[str, str]:
    """Create the user's two collections (+ payload indexes) if missing.

    Returns `(text_collection_name, image_collection_name)` for caller convenience.
    """
    text_name = user_text_collection(user)
    image_name = user_image_collection(user)
    await qdrant.ensure_collection(text_name, dim=TEXT_DIM)
    await qdrant.ensure_user_payload_indexes(text_name)
    await qdrant.ensure_collection(image_name, dim=IMAGE_DIM)
    await qdrant.ensure_user_payload_indexes(image_name)
    return text_name, image_name


__all__ = [
    "USER_TEXT_PREFIX", "USER_IMAGE_PREFIX",
    "sanitize_user", "user_text_collection", "user_image_collection",
    "ensure_user_collections",
]
