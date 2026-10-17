"""Public upload-layout labels retain complete reviewed evidence and source pages."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from audrey.kb.chunk import chunk_text
from audrey.kb.extract import extract_text

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_embedding_upload_fixture", ROOT / "scripts/analysis/build_embedding_upload_fixture.py",
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _normalized(text):
    return " ".join(text.split())


def test_frozen_upload_fixture_preserves_every_reviewed_question_and_quotation():
    reviewed_path = ROOT / "evals/cases/embedding_real_documents.json"
    reviewed = json.loads(reviewed_path.read_bytes())
    fixture = json.loads((ROOT / "evals/cases/embedding_upload_documents.json").read_bytes())
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert fixture["sources"] == reviewed["sources"]
    assert fixture["derivation"]["reviewed_fixture_sha256"] == hashlib.sha256(reviewed_path.read_bytes()).hexdigest()
    assert fixture["chunk_policy"]["tokens"] == config["kb"]["chunk_tokens"]
    assert fixture["chunk_policy"]["overlap_tokens"] == config["kb"]["chunk_overlap"]
    assert fixture["label_policy"]["original_reviewed_policy"] == reviewed["label_policy"]
    assert fixture["label_policy"]["new_independent_questions"] is False
    assert fixture["label_policy"]["new_model_calls_during_build"] == 0
    assert len(fixture["documents"]) == 168
    assert len(fixture["queries"]) == 80
    assert fixture["derivation"]["required_evidence_preserved"] == 96
    assert any(row["end_page"] > row["page"] for row in fixture["documents"])
    sources = {source["id"]: source for source in fixture["sources"]}
    for row in fixture["documents"]:
        assert row["workload"] == "documents"
        assert row["split"] == sources[row["source_id"]]["split"]
        assert 1 <= row["page"] <= row["end_page"] <= sources[row["source_id"]]["pages"]
        assert row["end_offset"] - row["offset"] == len(row["text"])
        assert row["text"].strip() == row["text"]
        assert row["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()
    for original, query in zip(reviewed["queries"], fixture["queries"], strict=True):
        for key in ("id", "text", "group", "split", "evidence", "kind", "answerable"):
            assert query[key] == original[key]
        assert query["reference_layout_relevant"] == original["relevant"]
        expected = set()
        for evidence in original["evidence"]:
            hits = {
                row["id"] for row in fixture["documents"]
                if row["source_id"] == evidence["source_id"]
                and row["page"] <= evidence["page"] <= row["end_page"]
                and _normalized(evidence["quote"]) in _normalized(row["text"])
            }
            assert hits, (query["id"], evidence)
            expected.update(hits)
        assert set(query["relevant"]) == expected
        assert bool(query["relevant"]) is query["answerable"]


def _pdf(path, pages):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.fixture
def public_inputs(tmp_path):
    quote = "The inspection day is Tuesday for this project."
    sources, queries = [], []
    for split in ("calibration", "holdout"):
        path = tmp_path / f"{split}.pdf"
        _pdf(path, [
            "Repeated filler line. " * 40 + quote + " More repeated filler. " * 12,
            "Second page introduction. " * 12 + quote + " A final explanatory paragraph. " * 30,
        ])
        sources.append({
            "id": split, "url": f"https://example.gov/{split}.pdf", "split": split,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": 2,
            "title": f"Frozen public {split}", "license": "US government work",
        })
        queries.extend([
            {"id": f"{split}-positive", "text": "What is the project inspection day?",
             "group": "reference", "split": split, "relevant": ["old-page-window"],
             "answerable": True, "kind": "paraphrase", "evidence": [
                 {"source_id": split, "page": 2, "quote": quote},
             ]},
            {"id": f"{split}-negative", "text": "What is the project budget?",
             "group": "reference", "split": split, "relevant": [],
             "answerable": False, "kind": "unsupported", "evidence": []},
        ])
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({
        "schema": 2, "language": "en", "public": True, "synthetic": False,
        "sources": sources, "queries": queries, "documents": [],
        "label_policy": {"retrospective": True},
    }))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"kb": {"chunk_tokens": 80, "chunk_overlap": 60}}))
    return reviewed, tmp_path, config


def test_builder_uses_real_upload_chunks_and_correct_page_occurrences(public_inputs):
    reviewed, directory, config = public_inputs
    fixture = builder.build_fixture(reviewed, directory, config)
    assert fixture == builder.build_fixture(reviewed, directory, config)
    for source in fixture["sources"]:
        raw = extract_text(directory / f"{source['id']}.pdf")
        actual = chunk_text(raw, chunk_tokens=80, overlap_tokens=60)
        rows = [row for row in fixture["documents"] if row["source_id"] == source["id"]]
        assert [row["text"] for row in rows] == [chunk.text for chunk in actual]
        for row in rows:
            assert raw[row["offset"]:row["end_offset"]] == row["text"]
        query = next(row for row in fixture["queries"] if row["id"] == f"{source['id']}-positive")
        quote = query["evidence"][0]["quote"]
        # The same complete sentence also appears on page1. Only its page2
        # occurrence is part of the pre-reviewed label for this question.
        quote_start = raw.rfind(quote)
        expected = {row["id"] for row in rows
                    if row["offset"] <= quote_start
                    and quote_start + len(quote) <= row["end_offset"]}
        assert len(expected) >= 2
        assert set(query["relevant"]) == expected
        assert any(quote in row["text"] and row["id"] not in expected for row in rows)


def test_changed_pdf_is_refused_before_any_extraction(public_inputs, monkeypatch):
    reviewed, directory, config = public_inputs
    (directory / "holdout.pdf").write_bytes(b"a changed source")

    def forbidden(_path):
        raise AssertionError("all checksums must be checked before extraction")

    monkeypatch.setattr(builder, "extract_text", forbidden)
    with pytest.raises(builder.FixtureBuildError, match="checksum"):
        builder.build_fixture(reviewed, directory, config)


@pytest.mark.parametrize("change, message", [
    (lambda evidence: evidence.update(page=3), "required source page"),
    (lambda evidence: evidence.update(quote="The inspection day was never recorded here."), "required source page"),
    (lambda evidence: evidence.update(quote="Tuesday"), "discriminating quotation"),
])
def test_missing_or_weak_original_evidence_is_never_silently_dropped(public_inputs, change, message):
    reviewed, directory, config = public_inputs
    body = json.loads(reviewed.read_bytes())
    change(body["queries"][0]["evidence"][0])
    reviewed.write_text(json.dumps(body))
    with pytest.raises(builder.FixtureBuildError, match=message):
        builder.build_fixture(reviewed, directory, config)


def test_required_quote_crossing_every_window_blocks_fixture(public_inputs):
    reviewed, directory, config = public_inputs
    config.write_text(yaml.safe_dump({"kb": {"chunk_tokens": 8, "chunk_overlap": 2}}))
    with pytest.raises(builder.FixtureBuildError, match="crosses every upload window"):
        builder.build_fixture(reviewed, directory, config)


def test_unrepresented_alternative_quote_blocks_instead_of_narrowing_labels(public_inputs):
    reviewed, directory, config = public_inputs
    body = json.loads(reviewed.read_bytes())
    alternate = copy.deepcopy(body["queries"][0]["evidence"][0])
    alternate["quote"] = "This required alternative was never present in the source."
    body["queries"][0]["evidence"].append(alternate)
    reviewed.write_text(json.dumps(body))
    with pytest.raises(builder.FixtureBuildError, match="required source page"):
        builder.build_fixture(reviewed, directory, config)
