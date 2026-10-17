#!/usr/bin/env python3
"""Remap the frozen public document labels to the production upload layout.

Only the four checksummed PDFs named by the reviewed fixture are read. This
does not download sources, contact models, read uploads or change any index.
The output reuses existing questions and retrospective annotations; it is a
layout extension, not a new independent document-quality benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import tiktoken
import yaml
from pypdf import PdfReader

from audrey.kb.chunk import chunk_text
from audrey.kb.extract import extract_text

ROOT = Path(__file__).resolve().parents[2]


class FixtureBuildError(ValueError):
    """Frozen input or answer evidence cannot be represented faithfully."""


def _policy(path: Path) -> tuple[int, int]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    kb = config.get("kb", {}) if isinstance(config, dict) else {}
    size, overlap = kb.get("chunk_tokens"), kb.get("chunk_overlap")
    if (isinstance(size, bool) or not isinstance(size, int) or size <= 0
            or isinstance(overlap, bool) or not isinstance(overlap, int)
            or not 0 <= overlap < size):
        raise FixtureBuildError("configuration must specify valid upload chunk size and overlap")
    return size, overlap


def _source_path(directory: Path, source: dict) -> Path:
    url = urlparse(source["url"])
    name = Path(url.path).name
    if url.scheme != "https" or not name.endswith(".pdf"):
        raise FixtureBuildError("public source must identify an HTTPS PDF")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise FixtureBuildError(f"cached public PDF is missing: {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise FixtureBuildError(f"cached public PDF checksum differs: {name}")
    return path


def _page_spans(path: Path, raw: str, expected_pages: int) -> list[tuple[int, int, int]]:
    pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    if len(pages) != expected_pages:
        raise FixtureBuildError("cached source page count differs from frozen metadata")
    if "\n\n".join(text for text in pages if text.strip()) != raw:
        raise FixtureBuildError("page provenance differs from production upload extraction")
    spans = []
    position = 0
    for number, text in enumerate(pages, 1):
        if not text.strip():
            continue
        if spans:
            position += 2
        spans.append((position, position + len(text), number))
        position += len(text)
    return spans


def _documents(source: dict, path: Path, size: int, overlap: int) -> tuple[list[dict], dict]:
    raw = extract_text(path)
    page_spans = _page_spans(path, raw, source["pages"])
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(raw.strip())
    leading = len(raw) - len(raw.lstrip())
    rows = []
    for chunk in chunk_text(raw, chunk_tokens=size, overlap_tokens=overlap):
        # Use the returned production chunk ordinal to recover its exact token
        # window. Searching by text would mislocate repeated headers/passages.
        token_start = chunk.idx * (size - overlap)
        try:
            window = encoder.decode(tokens[token_start:token_start + size], errors="strict")
            prefix = encoder.decode(tokens[:token_start], errors="strict")
        except UnicodeDecodeError as error:
            raise FixtureBuildError("upload token boundary cannot preserve source characters") from error
        offset = leading + len(prefix) + len(window) - len(window.lstrip())
        end = offset + len(chunk.text)
        if raw[offset:end] != chunk.text:
            raise FixtureBuildError("production upload chunk does not match its exact source offset")
        pages = [page for start, stop, page in page_spans if start < end and stop > offset]
        if not pages:
            raise FixtureBuildError("upload chunk has no source page")
        rows.append({
            "id": f"{source['id']}-upload-c{chunk.idx:04d}",
            "text": chunk.text, "source_id": source["id"],
            "page": pages[0], "end_page": pages[-1],
            "offset": offset, "end_offset": end,
            "split": source["split"], "workload": "documents",
            "text_sha256": hashlib.sha256(chunk.text.encode()).hexdigest(),
        })
    return rows, {"text": raw, "pages": page_spans}


def _quote_spans(evidence: dict, extracted: dict) -> list[tuple[int, int]]:
    words = evidence["quote"].split()
    if len(words) < 4:
        raise FixtureBuildError("answer labels require a discriminating quotation, not a keyword")
    expression = re.compile(r"\s+".join(re.escape(word) for word in words))
    spans = []
    for start, stop, page in extracted["pages"]:
        if page == evidence["page"]:
            spans.extend((start + match.start(), start + match.end())
                         for match in expression.finditer(extracted["text"][start:stop]))
    if not spans:
        raise FixtureBuildError("reviewed quotation is absent from its required source page")
    return spans


def build_fixture(reviewed_path: Path, source_dir: Path, config_path: Path) -> dict:
    original_bytes = reviewed_path.read_bytes()
    original = json.loads(original_bytes)
    if (original.get("schema") != 2 or original.get("language") != "en"
            or original.get("public") is not True or original.get("synthetic") is not False):
        raise FixtureBuildError("source labels must be frozen public English document evidence")
    size, overlap = _policy(config_path)
    # Check every source before parsing any PDF, so a changed file never enters
    # a partly rebuilt fixture. No URL is fetched by this builder.
    paths = {source["id"]: _source_path(source_dir, source) for source in original["sources"]}
    extracted, documents = {}, []
    for source in original["sources"]:
        rows, provenance = _documents(source, paths[source["id"]], size, overlap)
        documents.extend(rows)
        extracted[source["id"]] = provenance
    queries = []
    for original_query in original["queries"]:
        query = copy.deepcopy(original_query)
        query["reference_layout_relevant"] = query["relevant"]
        query["relevant"] = []
        query["workload"] = "documents"
        matches = set()
        if original_query["answerable"] and not original_query.get("evidence"):
            raise FixtureBuildError("answerable question has no reviewed source quotation")
        for evidence in original_query.get("evidence", []):
            source_id = evidence["source_id"]
            spans = _quote_spans(evidence, extracted[source_id])
            containing = [row for row in documents if row["source_id"] == source_id
                          and row["split"] == query["split"]
                          and any(row["offset"] <= start and stop <= row["end_offset"]
                                  for start, stop in spans)]
            if not containing:
                raise FixtureBuildError(
                    f"required evidence crosses every upload window: {query['id']} page {evidence['page']}"
                )
            matches.update(row["id"] for row in containing)
        query["relevant"] = [row["id"] for row in documents if row["id"] in matches]
        if bool(query["relevant"]) is not original_query["answerable"]:
            raise FixtureBuildError("remapped answerability differs from the reviewed question")
        query["relevance_basis"] = (
            "Every production upload window fully containing any previously reviewed "
            "answer quotation at its required source page; no keyword-only labels."
        )
        queries.append(query)
    return {
        "schema": 2, "language": "en", "public": True, "synthetic": False,
        "workload": "documents",
        "scope": "Production upload-layout extension of frozen public IRS source-reading cases; not independent new questions or tax advice",
        "sources": copy.deepcopy(original["sources"]),
        "chunk_policy": {
            "extraction": "audrey.kb.extract.extract_text",
            "implementation": "audrey.kb.chunk.chunk_text",
            "tokens": size, "overlap_tokens": overlap, "tokenizer": "cl100k_base",
            "page_preserving": False,
            "offset_basis": "character offsets in the complete production extracted document",
            "page_span": "first/last PDF source page overlapped by the upload window",
        },
        "label_policy": {
            "original_reviewed_policy": copy.deepcopy(original["label_policy"]),
            "layout_labels_frozen_before_new_model_calls": True,
            "new_independent_questions": False,
            "new_model_calls_during_build": 0,
            "all_original_evidence_required": True,
            "limitation": "Existing alternative labels were retrospectively reviewed; changing chunk layout does not create an independent annotation or source holdout",
        },
        "derivation": {
            "reviewed_fixture": reviewed_path.name,
            "reviewed_fixture_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "questions_preserved": len(queries),
            "required_evidence_preserved": sum(len(query.get("evidence", [])) for query in queries),
            "source_bytes_verified": True,
        },
        "documents": documents, "queries": queries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviewed-cases", type=Path, default=ROOT / "evals/cases/embedding_real_documents.json")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = build_fixture(args.reviewed_cases, args.source_dir, args.config)
    args.output.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "documents": len(fixture["documents"]),
                      "queries": len(fixture["queries"]),
                      "required_evidence": fixture["derivation"]["required_evidence_preserved"],
                      "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
