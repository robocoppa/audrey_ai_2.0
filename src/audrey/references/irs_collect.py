"""Collect public IRS catalogs and retain original PDFs without model calls.

Run with python -m audrey.references.irs_collect --root PATH. This collector
does not read uploads, credentials, application configuration or vector stores.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from audrey.references.irs_archive import ArchiveStore
from audrey.references.irs_catalog import CatalogParseError, normalize_irs_url, parse_catalog_page

SEEDS = {
    "current": "https://www.irs.gov/forms-instructions-and-publications?items_per_page=200",
    "prior": "https://www.irs.gov/prior-year-forms-and-instructions?items_per_page=200",
    "bulletins": "https://www.irs.gov/irb",
}
MIB = 1024 * 1024
GIB = 1024 * MIB


class CollectionError(Exception):
    """A bounded public collection operation failed."""


class CapacityError(CollectionError):
    """Storage or the configured byte budget cannot hold another document."""


class PublicIRSClient:
    """Rate-limited HTTPS reads with validated redirects and bounded responses."""

    def __init__(
        self, *, delay: float = 1.0, transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.delay = delay
        self._last_request = 0.0
        self.client = httpx.Client(
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={"User-Agent": "Audrey-Reference-Archive/1.0 (public IRS documents)"},
        )

    def close(self) -> None:
        self.client.close()

    @contextlib.contextmanager
    def response(self, url: str, *, headers: dict[str, str] | None = None):
        target = normalize_irs_url(url)
        expected_pdf = urlsplit(target).path.endswith(".pdf")
        for attempt in range(3):
            try:
                for _ in range(6):
                    pause = self.delay - (time.monotonic() - self._last_request)
                    if pause > 0:
                        time.sleep(pause)
                    self._last_request = time.monotonic()
                    request = self.client.build_request("GET", target, headers=headers or {})
                    response = self.client.send(request, stream=True)
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        response.close()
                        if not location:
                            raise CollectionError("IRS redirect had no location")
                        target = normalize_irs_url(location, base_url=target)
                        if urlsplit(target).path.endswith(".pdf") != expected_pdf:
                            raise CollectionError("IRS redirect changed the resource type")
                        continue
                    break
                else:
                    raise CollectionError("IRS redirect limit exceeded")
                if response.status_code == 429 or response.status_code >= 500:
                    retry_header = response.headers.get("retry-after", "")
                    response.close()
                    if attempt == 2:
                        raise CollectionError("IRS request remained rate-limited or unavailable")
                    try:
                        retry_delay = min(60.0, max(1.0, float(retry_header)))
                    except ValueError:
                        retry_delay = 2.0 ** (attempt + 1)
                    time.sleep(retry_delay)
                    continue
                if response.status_code not in (200, 304):
                    code = response.status_code
                    response.close()
                    raise CollectionError(f"IRS request returned HTTP {code}")
            except httpx.HTTPError as error:
                if attempt == 2:
                    raise CollectionError(f"IRS connection failed: {type(error).__name__}") from None
                time.sleep(2.0 ** (attempt + 1))
                continue
            # Caller errors cannot make this contextmanager yield twice.
            try:
                yield response
            finally:
                response.close()
            return
        raise CollectionError("IRS request could not complete")

    def catalog(self, url: str) -> tuple[bytes, str]:
        with self.response(url) as response:
            if response.status_code != 200:
                raise CollectionError("Catalog response did not contain a document")
            content = bytearray()
            start = time.monotonic()
            for block in response.iter_bytes():
                if len(content) + len(block) > 4 * MIB:
                    raise CollectionError("Catalog exceeded the 4 MiB limit")
                if time.monotonic() - start > 90:
                    raise CollectionError("Catalog response exceeded the time limit")
                content.extend(block)
            return bytes(content), str(response.url)

    def pdf(
        self, url: str, destination: Path, *, headers: dict[str, str],
        max_bytes: int, min_free_bytes: int,
    ) -> dict[str, Any]:
        with self.response(url, headers=headers) as response:
            metadata = {
                "final_url": str(response.url), "etag": response.headers.get("etag", ""),
                "last_modified": response.headers.get("last-modified", ""),
            }
            if response.status_code == 304:
                return {**metadata, "not_modified": True, "bytes": 0}
            length = response.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_bytes:
                raise CapacityError("PDF exceeds the file or remaining download-byte limit")
            start = time.monotonic()
            size = 0
            with destination.open("wb") as output:
                for block in response.iter_bytes(chunk_size=64 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise CapacityError("PDF exceeds the file or remaining download-byte limit")
                    if time.monotonic() - start > 120:
                        raise CollectionError("PDF response exceeded the time limit")
                    if shutil.disk_usage(destination.parent).free < min_free_bytes + len(block):
                        raise CapacityError("Available disk space reached the configured reserve")
                    output.write(block)
            with destination.open("rb") as source:
                if source.read(5) != b"%PDF-":
                    raise CollectionError("IRS download is not a PDF")
                source.seek(max(0, size - 65536))
                if b"%%EOF" not in source.read():
                    raise CollectionError("IRS PDF appears truncated (missing end marker)")
            return {**metadata, "not_modified": False, "bytes": size}


def _verified_local(root: Path, revision: dict[str, Any] | None) -> bool:
    if not revision:
        return False
    path = root / revision["relative_path"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        return False
    if path.stat().st_size != revision["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest() == revision["sha256"]


def _emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def collect(args: argparse.Namespace, store: ArchiveStore, client: PublicIRSClient) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    run = {"attempted": 0, "downloaded": 0, "cached": 0, "unchanged": 0, "new_bytes": 0}
    limits: list[str] = []
    verified: set[str] = set()
    for catalog in args.catalogs:
        queue = deque([SEEDS[catalog]])
        visited: set[str] = set()
        rows = 0
        expected_total: int | None = None
        catalog_failed = False
        while queue:
            if args.max_pages is not None and len(visited) >= args.max_pages:
                limits.append(f"{catalog}: requested catalog page limit")
                break
            url = queue.popleft()
            if url in visited:
                continue
            try:
                body, final_url = client.catalog(url)
                page = parse_catalog_page(body.decode("utf-8"), final_url, catalog)
                if shutil.disk_usage(store.root).free < args.min_free_bytes + 2 * len(body):
                    raise CapacityError("Catalog storage reached the configured free-space reserve")
                store.record_catalog_page(
                    catalog, final_url, body, page.entries,
                    next_urls=page.next_urls, total_rows=page.total_rows,
                )
                visited.add(url)
                rows += page.row_count
                if page.total_rows is not None:
                    if expected_total is not None and expected_total != page.total_rows:
                        raise CollectionError("Catalog changed during enumeration; rerun to establish coverage")
                    expected_total = page.total_rows
                queue.extend(next_url for next_url in page.next_urls if next_url not in visited)
                _emit("catalog_page", catalog=catalog, pages=len(visited), rows=rows)
            except (CatalogParseError, CollectionError, UnicodeError, httpx.HTTPError) as error:
                catalog_failed = True
                failures.append({"catalog": catalog, "url": url, "error": str(error)})
                _emit("catalog_failed", catalog=catalog, error=str(error))
                break
        complete = not queue and not catalog_failed
        if complete and expected_total is not None and rows != expected_total:
            complete = False
            failures.append({
                "catalog": catalog, "error": "Observed catalog row count does not match official total",
                "url": SEEDS[catalog],
            })
        coverage[catalog] = {
            "pages": len(visited), "rows_seen": rows,
            "official_total_rows": expected_total, "enumeration_complete": complete,
        }

    if not args.discover_only:
        priorities = {catalog: index for index, catalog in enumerate(SEEDS) if catalog in args.catalogs}
        documents = sorted(store.documents(args.catalogs), key=lambda item: (
            min(priorities[name] for name in item["catalogs"] if name in priorities),
            item["source_url"],
        ))
        for document in documents:
            if not document["eligible"]:
                continue
            url = document["source_url"]
            revision = store.latest_revision(url)
            valid_local = _verified_local(store.root, revision)
            metadata_changed = bool(revision and any(
                document[field] != revision["metadata"].get(field, "")
                for field in ("revision_date", "posted_date")
            ))
            refresh = args.refresh or metadata_changed
            if valid_local and not refresh:
                if document.get("error"):
                    store.record_not_modified(url)
                verified.add(url)
                run["cached"] += 1
                continue
            if args.max_downloads is not None and run["attempted"] >= args.max_downloads:
                limits.append("requested PDF download-attempt limit")
                break
            if run["new_bytes"] >= args.max_new_bytes:
                failures.append({"url": url, "error": "Configured new-download byte budget reached"})
                limits.append("download-byte budget")
                break
            if shutil.disk_usage(store.root).free < args.min_free_bytes:
                failures.append({"url": url, "error": "Available disk space reached the configured reserve"})
                limits.append("free-space reserve")
                break
            run["attempted"] += 1
            headers: dict[str, str] = {}
            if valid_local and refresh and revision:
                if revision.get("etag"):
                    headers["If-None-Match"] = revision["etag"]
                elif revision.get("last_modified"):
                    headers["If-Modified-Since"] = revision["last_modified"]
            descriptor, name = tempfile.mkstemp(prefix="irs-", suffix=".part", dir=store.root / ".staging")
            os.close(descriptor)
            temporary = Path(name)
            try:
                metadata = client.pdf(
                    url, temporary, headers=headers,
                    max_bytes=min(args.max_file_bytes, args.max_new_bytes - run["new_bytes"]),
                    min_free_bytes=args.min_free_bytes,
                )
                if metadata["not_modified"]:
                    if not valid_local:
                        raise CollectionError("IRS returned not-modified without a verified local PDF")
                    store.record_not_modified(url)
                    run["unchanged"] += 1
                else:
                    if shutil.disk_usage(store.root).free < args.min_free_bytes + metadata["bytes"]:
                        raise CapacityError("Archiving the verified PDF would cross the free-space reserve")
                    store.record_download(
                        url, temporary, metadata["final_url"],
                        etag=metadata["etag"], last_modified=metadata["last_modified"],
                    )
                    run["downloaded"] += 1
                    run["new_bytes"] += metadata["bytes"]
                verified.add(url)
                _emit("pdf_progress", **run)
            except (CollectionError, CatalogParseError, httpx.HTTPError, OSError, ValueError) as error:
                safe_error = str(error) if not isinstance(error, httpx.HTTPError) else type(error).__name__
                store.record_failure(url, safe_error)
                failures.append({"url": url, "error": safe_error})
                _emit("pdf_failed", url=url, error=safe_error)
                if isinstance(error, CapacityError):
                    limits.append("file size, byte budget or free-space limit")
                    break
            finally:
                temporary.unlink(missing_ok=True)

    inventory = store.summary()
    remaining = sum(
        1 for doc in store.documents(args.catalogs)
        if doc["eligible"] and doc["source_url"] not in verified
    ) if not args.discover_only else inventory["eligible"]
    catalogs_complete = all(item["enumeration_complete"] for item in coverage.values())
    complete = catalogs_complete and remaining == 0 and not failures and not args.discover_only
    status = "completed" if complete else ("findings" if failures else "partial")
    store.export_manifest()
    return {
        "schema": 1, "status": status, "created_at": datetime.now(UTC).isoformat(),
        "archive_root": str(store.root), "selected_catalogs": args.catalogs,
        "coverage": coverage, "archive_complete_for_selected_catalogs": complete,
        "discovery_only": args.discover_only, "run": run, "inventory": inventory,
        "remaining_known_documents": remaining, "limits": limits, "failures": failures,
        "scope": "Official English IRS catalog PDFs; not every source of federal tax law",
        "tax_year_applicability": "Not inferred from catalog revision dates; requires source verification",
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", required=True, type=Path)
    result.add_argument("--catalogs", default="current,prior,bulletins")
    result.add_argument("--max-pages", type=_positive_int, help="Catalog pages per catalog (bounded check)")
    result.add_argument("--max-downloads", type=_positive_int, help="PDF attempts this run (resume later)")
    result.add_argument("--max-new-bytes", type=_positive_int, default=10 * GIB)
    result.add_argument("--min-free-bytes", type=_positive_int, default=GIB)
    result.add_argument("--max-file-bytes", type=_positive_int, default=128 * MIB)
    result.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between IRS requests")
    result.add_argument("--refresh", action="store_true", help="Conditionally check verified PDFs for updates")
    result.add_argument("--discover-only", action="store_true")
    result.add_argument("--report", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    args.catalogs = list(dict.fromkeys(args.catalogs.split(",")))
    if not args.catalogs or any(catalog not in SEEDS for catalog in args.catalogs):
        parser().error("--catalogs must be a comma-separated selection of current,prior,bulletins")
    if not 0.25 <= args.delay <= 60:
        parser().error("--delay must be between 0.25 and 60 seconds")
    report: dict[str, Any] = {"schema": 1, "status": "setup_error"}
    code = 2
    started = False
    try:
        args.root.mkdir(parents=True, exist_ok=True)
        if args.root.is_symlink():
            raise ValueError("Archive root must not be a symlink")
        with (args.root / ".collect.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another collector is using this archive") from None
            with ArchiveStore(args.root) as store:
                client = PublicIRSClient(delay=args.delay)
                try:
                    started = True
                    report = collect(args, store, client)
                    code = 1 if report["failures"] else 0
                finally:
                    client.close()
    except KeyboardInterrupt:
        report = {"schema": 1, "status": "interrupted", "resume": "Run the same command again"}
        code = 130
    except (OSError, ValueError) as error:
        code = 3 if started else 2
        report = {
            "schema": 1, "status": "crashed" if started else "setup_error", "error": str(error),
        }
    except Exception as error:  # noqa: BLE001 — operator CLI always emits a final public report
        report = {"schema": 1, "status": "crashed", "error": type(error).__name__}
        code = 3
    report["exit_code"] = code
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(args.report.suffix + ".part")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.report)
        except OSError:
            print("Could not write collector report", flush=True)
            code = 3
    print(json.dumps(report, indent=2), flush=True)
    return code


def _interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())

