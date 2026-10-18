"""Operator entry point for the isolated public-reference index job."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import tempfile
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from audrey.config import get_config
from audrey.embedding_profile import profile_from_mapping
from audrey.kb.qdrant import QdrantKB
from audrey.models.ollama import OllamaClient
from audrey.references.embedding import reference_embedder
from audrey.references.indexing import build_reference_index
from audrey.references.library import ReferenceLibraryError, ReferenceLibraryRegistry
from audrey.references.releases import ReleaseError, ReleaseStore, validate_extraction_gaps


async def _run(args, cfg, registry, releases):
    ollama = OllamaClient(cfg.env.ollama_host)
    qdrant = QdrantKB(cfg.env.qdrant_host, cfg.env.qdrant_port, embedding_profile=profile_from_mapping(cfg.raw["text_embedding"]))
    embedder = reference_embedder(ollama, cfg.raw)
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        return await build_reference_index(
            registry,
            releases,
            qdrant,
            embedder,
            library_id=args.library,
            catalog=args.catalog,
            max_documents=args.max_documents,
            product_numbers=args.product_number,
            release_id=args.release_id,
            rebuild=args.rebuild,
            allowed_extraction_gaps=args.allow_extraction_gap,
            progress=lambda event: print(json.dumps(event), flush=True),
        )
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
        await ollama.aclose()
        qdrant.close()


def write_report(path: Path, report: dict) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Report path must not be a symbolic link")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def private_diagnostic(root: Path, error: BaseException) -> str:
    """Retain traceback on Tower without traversing a diagnostic-path symlink."""
    root = Path(root).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Diagnostic root must be an absolute server path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root.anchor, flags)
    try:
        # Anchor every component by descriptor: checking only the last component
        # would still follow a symbolic link in an ancestor of the index root.
        for component in root.parts[1:]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            child_fd = os.open(component, flags, dir_fd=root_fd)
            os.close(root_fd)
            root_fd = child_fd
        try:
            os.mkdir("diagnostics", mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        directory_fd = os.open("diagnostics", flags, dir_fd=root_fd)
        try:
            os.fchmod(directory_fd, 0o700)
            name = f"failure-{uuid.uuid4().hex}.traceback.txt"
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                os.fchmod(output.fileno(), 0o600)
                traceback.print_exception(error, file=output)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(directory_fd)
            return f"diagnostics/{name}"
        finally:
            os.close(directory_fd)
    finally:
        os.close(root_fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="irs-federal")
    parser.add_argument("--catalog", choices=["current", "prior", "bulletins"], default="current")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--product-number", action="append", default=[])
    parser.add_argument("--release-id")
    parser.add_argument(
        "--allow-extraction-gap", action="append", default=[], metavar="DOCUMENT_ID",
        help="Allow one explicitly approved failed document in an existing release; repeat for exact IDs",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Build a replacement even when sources are unchanged"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    started = monotonic()
    report = {"schema": 1, "status": "setup_error"}
    code = 2
    index_root = None
    try:
        args.allow_extraction_gap = validate_extraction_gaps(args.allow_extraction_gap)
        if args.allow_extraction_gap and (args.release_id is None or args.rebuild):
            raise ValueError("Extraction gaps require an explicit existing release ID without rebuild")
        cfg = get_config()
        reference_cfg = cfg.raw.get("reference_libraries", {})
        index_root = Path(reference_cfg.get("index_root", "/data/reference-libraries"))
        registry = ReferenceLibraryRegistry(reference_cfg)
        # Source preflight occurs before opening network clients.
        registry.catalog(args.library, catalog=args.catalog, limit=1)
        releases = ReleaseStore(index_root)
        if not args.report.is_absolute() or args.report.parent != releases.root / "reports":
            raise ValueError("Report must be in the public index state's reports directory")
        # An operator override of the container name cannot create two writers.
        lock_path = releases.root / ".index.lock"
        with os.fdopen(
            os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a"
        ) as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another public reference index build is running") from None
            try:
                report = asyncio.run(_run(args, cfg, registry, releases))
                code = 0 if report["status"] == "published" else 1
            except (asyncio.CancelledError, KeyboardInterrupt):
                report, code = {"schema": 1, "status": "interrupted"}, 130
    except (ReferenceLibraryError, ReleaseError, ValueError) as exc:
        report["error"] = str(exc)[:300]
    except Exception as exc:  # noqa: BLE001 — do not print connection strings or credentials
        report, code = {"schema": 1, "status": "crashed", "error": type(exc).__name__}, 3
        try:
            if index_root is None:
                raise ValueError("Index root is not configured")
            report["private_diagnostic"] = private_diagnostic(index_root, exc)
        except Exception:  # noqa: BLE001 — diagnostics must not replace the original failure
            report["private_diagnostic_saved"] = False
    report.update(
        exit_code=code,
        created_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=round(monotonic() - started, 3),
    )
    try:
        write_report(args.report, report)
    except (OSError, ValueError):
        print(
            json.dumps({"schema": 1, "status": "report_write_failed", "exit_code": 3}), flush=True
        )
        return 3
    print(json.dumps(report, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
