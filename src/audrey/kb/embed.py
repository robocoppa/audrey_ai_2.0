"""Embedding providers for the KB.

Two flavors:
  - Text  → `nomic-embed-text` via Ollama's `/api/embed` (768-d, cosine).
  - Image → CLIP ViT-B-32 via sentence-transformers (512-d, cosine).

Text embeddings are async-native; the CLIP model is a sync torch pipeline
wrapped in `asyncio.to_thread`. The CLIP model weights are ~380 MB and
cache to `/root/.cache/clip` (bind-mounted on Unraid).

Both embedders normalize outputs to unit length — Qdrant cosine search
on a unit-vector index is equivalent to dot-product, which is what Qdrant
actually uses internally for `Distance.COSINE`, but we normalize anyway
so the same vectors work if someone switches the collection to DOT.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import logging
import math
import socket
import urllib.parse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from audrey.models.ollama import OllamaClient

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


# ── SSRF guards (Phase 27) ────────────────────────────────────────────
#
# `_fetch_image` is called by the public `/v1/kb/query/image` route with
# whatever URL an authenticated user supplied. Without these guards a
# user could:
#   - Probe internal services via redirects ("does qdrant:6333 respond?")
#   - Hit Audrey's own loopback metrics/admin endpoints
#   - OOM the container with a multi-GB response (no streaming cap)
#
# Phase 26 already requires auth on the chat completions route that
# eventually triggers tool calls; Phase 27 adds defense-in-depth at the
# fetch layer itself. Trade-offs documented inline.

_IMAGE_FETCH_BYTE_CAP = 25 * 1024 * 1024  # 25 MB; huge for an image
_ALLOWED_IMAGE_SCHEMES = frozenset({"https"})
_ALLOWED_CONTENT_TYPE_PREFIX = "image/"


def _is_unsafe_address(host: str) -> bool:
    """True if any DNS-resolved IP for `host` is private / loopback / etc.

    Resolves via `socket.getaddrinfo` (IPv4 + IPv6). Any single resolved
    address falling into a non-public range fails the host. Unresolvable
    hosts also fail (better to reject than try and possibly hit something
    weird via dns search domains).

    Doesn't defend against DNS rebinding — the actual httpx connection
    re-resolves and could land on a different IP. Real mitigation would
    require connecting to the resolved IP and passing the hostname via
    SNI. Out of scope for Phase 27; bounded by Phase 26 requiring auth.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for _, _, _, _, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return True
    return False


def _validate_image_url(url: str) -> None:
    """Raise ValueError if the URL is unsafe to fetch as an image."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_IMAGE_SCHEMES:
        raise ValueError(
            f"image_url scheme must be one of {sorted(_ALLOWED_IMAGE_SCHEMES)}; got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError("image_url has no host")
    if _is_unsafe_address(parsed.hostname):
        raise ValueError(
            f"image_url host {parsed.hostname!r} resolves to a private, loopback, "
            "link-local, or otherwise non-public address"
        )

log = logging.getLogger(__name__)


# ─── Text embedder (ollama) ───────────────────────────────────────────

@dataclass(slots=True)
class TextEmbedder:
    ollama: OllamaClient
    model: str = "nomic-embed-text"
    timeout_s: float = 60.0
    batch_size: int = 64

    async def embed_one(self, text: str) -> list[float]:
        out = await self.embed_many([text])
        return out[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            got = await self.ollama.embed(model=self.model, texts=batch, timeout_s=self.timeout_s)
            vectors.extend(_normalize(v) for v in got)
        return vectors


# ─── Image embedder (CLIP) ────────────────────────────────────────────

@dataclass(slots=True)
class ImageEmbedder:
    model_name: str = "clip-ViT-B-32"
    cache_folder: str | None = None

    async def embed_url(self, url: str) -> list[float]:
        img = await _fetch_image(url)
        return await self.embed_pil(img)

    async def embed_b64(self, b64: str) -> list[float]:
        data = base64.b64decode(b64)
        img = await asyncio.to_thread(_pil_from_bytes, data)
        return await self.embed_pil(img)

    async def embed_path(self, path: str | Path) -> list[float]:
        img = await asyncio.to_thread(_pil_from_path, Path(path))
        return await self.embed_pil(img)

    async def embed_pil(self, image: "PILImage") -> list[float]:
        model = _load_clip(self.model_name, self.cache_folder)
        vec = await asyncio.to_thread(_clip_encode, model, image)
        return _normalize(vec)

    async def embed_text(self, text: str) -> list[float]:
        # CLIP's text and image encoders share the same 512-d embedding space,
        # so a text vector can be cosine-searched against `kb_images` directly.
        model = _load_clip(self.model_name, self.cache_folder)
        vec = await asyncio.to_thread(_clip_encode_text, model, text)
        return _normalize(vec)


# ─── Helpers ──────────────────────────────────────────────────────────

def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


async def _fetch_image(url: str) -> "PILImage":
    # Phase 27: validate URL before any I/O. Raises ValueError on bad
    # scheme, missing host, or host resolving to a private/loopback IP.
    # Run in a thread because socket.getaddrinfo can block.
    await asyncio.to_thread(_validate_image_url, url)

    # Wikimedia (and a handful of other CDNs) reject the default
    # `python-httpx/x.y.z` UA with 403. A normal browser UA gets through.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
    }
    # Phase 27: follow_redirects=False — a permitted public host could
    # 302 to an internal address otherwise. Stream the response with a
    # byte cap so a malicious server can't OOM us with a giant payload.
    async with httpx.AsyncClient(
        timeout=20.0, follow_redirects=False, headers=headers,
    ) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if not ctype.startswith(_ALLOWED_CONTENT_TYPE_PREFIX):
                raise ValueError(
                    f"image_url returned content-type {ctype!r}; expected image/*"
                )
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > _IMAGE_FETCH_BYTE_CAP:
                    raise ValueError(
                        f"image_url response exceeds {_IMAGE_FETCH_BYTE_CAP}-byte cap"
                    )
            return await asyncio.to_thread(_pil_from_bytes, bytes(buf))


def _pil_from_bytes(data: bytes) -> "PILImage":
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def _pil_from_path(path: Path) -> "PILImage":
    from PIL import Image

    img = Image.open(path)
    img.load()
    return img.convert("RGB")


@lru_cache(maxsize=2)
def _load_clip(model_name: str, cache_folder: str | None):
    from sentence_transformers import SentenceTransformer

    log.info("clip: loading %s (cache=%s)", model_name, cache_folder or "default")
    return SentenceTransformer(model_name, cache_folder=cache_folder)


def _clip_encode(model, image: "PILImage") -> list[float]:
    # sentence-transformers returns a numpy array; convert to plain list so
    # qdrant-client's JSON serializer is happy.
    out = model.encode([image], convert_to_numpy=True, normalize_embeddings=False)
    return [float(x) for x in out[0].tolist()]


def _clip_encode_text(model, text: str) -> list[float]:
    out = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
    return [float(x) for x in out[0].tolist()]


__all__ = ["TextEmbedder", "ImageEmbedder"]
