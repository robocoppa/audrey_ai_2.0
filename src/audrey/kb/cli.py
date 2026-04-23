"""KB ingest CLI.

    audrey-ingest [PATH ...]              — ingest one or more paths
    audrey-ingest --stats                 — print per-collection counts
    audrey-ingest --purge SOURCE          — remove a source from both collections

If no paths are given, falls back to `kb.dataset_paths` from config. Run
inside the audrey-ai container where Qdrant + Ollama are reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from audrey.config import get_config
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.ingest import ingest_many
from audrey.kb.qdrant import QdrantKB, normalize_source
from audrey.models.ollama import OllamaClient

log = logging.getLogger("audrey.kb.cli")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audrey-ingest", description=__doc__)
    p.add_argument("paths", nargs="*", help="Paths to ingest (defaults to config kb.dataset_paths)")
    p.add_argument("--stats", action="store_true", help="Print collection counts and exit")
    p.add_argument("--purge", metavar="SOURCE", help="Delete all points for a given source path")
    p.add_argument("--no-images", action="store_true", help="Skip image files (text-only ingest)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    kb_cfg = cfg.raw.get("kb", {}) or {}

    qdrant = QdrantKB(
        host=cfg.env.qdrant_host,
        port=cfg.env.qdrant_port,
        text_collection=kb_cfg.get("text_collection", "kb_text"),
        image_collection=kb_cfg.get("image_collection", "kb_images"),
    )
    await qdrant.ensure_collections()

    if args.stats:
        counts = await qdrant.counts()
        for name, n in counts.items():
            print(f"{name}: {n}")
        qdrant.close()
        return 0

    if args.purge:
        source = normalize_source(args.purge)
        await qdrant.delete_by_source(source, collection=qdrant.text_collection)
        await qdrant.delete_by_source(source, collection=qdrant.image_collection)
        print(f"purged: {source}")
        qdrant.close()
        return 0

    ollama = OllamaClient(cfg.env.ollama_host, default_timeout_s=float(cfg.timeouts.get("medium", 180)))
    text_embedder = TextEmbedder(ollama=ollama, model=kb_cfg.get("text_embedder", "nomic-embed-text"))
    image_embedder: ImageEmbedder | None = None
    if not args.no_images:
        image_embedder = ImageEmbedder(
            model_name=kb_cfg.get("image_model", "clip-ViT-B-32"),
            cache_folder="/root/.cache/clip",
        )

    roots = [Path(p) for p in (args.paths or kb_cfg.get("dataset_paths") or [])]
    if not roots:
        print("no paths given and kb.dataset_paths is empty", file=sys.stderr)
        qdrant.close()
        await ollama.aclose()
        return 2

    try:
        stats = await ingest_many(
            roots, qdrant=qdrant,
            text_embedder=text_embedder, image_embedder=image_embedder,
            chunk_tokens=int(kb_cfg.get("chunk_tokens", 1000)),
            overlap_tokens=int(kb_cfg.get("chunk_overlap", 100)),
        )
    finally:
        qdrant.close()
        await ollama.aclose()

    d = stats.as_dict()
    print(
        f"ingest complete: seen={d['files_seen']} text={d['files_text']}({d['chunks_text']} chunks) "
        f"images={d['files_image']} skipped={d['files_skipped']} errors={len(d['errors'])}"
    )
    if d["errors"]:
        for e in d["errors"][:10]:
            print(f"  error: {e}", file=sys.stderr)
    return 0 if not d["errors"] else 1


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
