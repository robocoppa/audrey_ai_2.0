"""Hermetic Japanese policy boundaries informed by the exposed prospective failure.

These are regression examples, not fresh prospective or human-reviewed controls.
The frozen case fixture is never loaded. Every router response is provided by
httpx.MockTransport; no model, live service, or production selector is involved.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "skill_selection_japanese_eval", _ROOT / "evals/eval_skill_selection.py"
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

DOCUMENT = "grounded-document-analysis"
VIDEO = "video-analysis"
_DOCUMENT = probe.FileEvidence("dune_notes.pdf", "document", "ready")
_OTHER_DOCUMENT = probe.FileEvidence("other_brief.pdf", "document", "ready")
_VIDEO = probe.FileEvidence("demo.mp4", "video", "ready")

# Literal reported prompt; deliberately independent of the frozen fixture file.
_EXPOSED_PROMPT = (
    "添付の dune_notes.pdf は読まずに、"
    "「打ち合わせは明日に変更しました」という文を丁寧な日本語に直してください。"
)


def _case(prompt, *, files=(_DOCUMENT,), mode="auto", expected="none"):
    return probe.Case(
        id="japanese-regression",
        category="ordinary" if expected == "none" else "positive",
        prompt=prompt,
        mode=mode,
        files=files,
        expected=expected,
        reason="Failure-informed hermetic boundary example.",
    )


def _evaluate_with_mock(case, *, backend, selected=DOCUMENT, repeats=1):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"skill_id": selected}),
                },
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://mock-ollama:11434",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await probe.evaluate(
                [case],
                backend=backend,
                client=client,
                model="qwen3.5:4b",
                repeats=repeats,
            )

    return asyncio.run(run()), requests


@pytest.mark.parametrize("mode", ["auto", "fast", "deep"])
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_exposed_japanese_rewrite_is_terminal_and_never_calls_router(mode, backend):
    case = _case(_EXPOSED_PROMPT, mode=mode)
    decision = probe.select_rules(case)
    assert decision.valid
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert decision.policy_reason
    assert probe.eligible_skills(case) == set()

    # A deliberately activating mock makes an accidental call observable.
    report, requests = _evaluate_with_mock(case, backend=backend, repeats=3)
    assert requests == []
    samples = report["backends"][backend]["samples"]
    assert len(samples) == 3
    for sample in samples:
        assert sample["selected"] == "none"
        assert sample["valid"] is True
        assert sample["source"] == "guard"
        assert sample["model_called"] is False
        assert sample["terminal_abstention"] is True
        assert sample["policy_reason"]
    assert report["selection_check"]["status"] == "matched_labels"


@pytest.mark.parametrize(
    "denial", ["読まずに", "読まないで", "開かずに", "開かないで", "参照せずに"]
)
@pytest.mark.parametrize("quotation", [("「", "」"), ("『", "』")])
def test_explicit_access_denial_with_supplied_text_is_terminal(denial, quotation):
    opening, closing = quotation
    case = _case(
        f"dune_notes.pdf は{denial}、"
        f"{opening}会議は延期です{closing}という文を丁寧に書き換えてください。"
    )
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert probe.eligible_skills(case) == set()


@pytest.mark.parametrize(
    "prompt",
    [
        "「dune_notes.pdf」は読まずに、「会議は延期です」という文を丁寧に直してください。",
        "『dune_notes.pdf』を開かないで、『会議は延期です』という文を丁寧に直してください。",
        "dune_notes.pdf は参照せずに、「dune_notes.pdf を分析してください」という文を英語に訳してください。",
    ],
)
def test_quoted_filename_or_instruction_does_not_override_real_access_denial(prompt):
    case = _case(prompt)
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert probe.eligible_skills(case) == set()


_POSITIVE_NEIGHBORS = [
    pytest.param(
        "私がアップロードした dune_notes.pdf の二つの案の評価を説明し、根拠の箇所を示してください。",
        (_DOCUMENT,), DOCUMENT, id="document-content",
    ),
    pytest.param(
        "私がアップロードした demo.mp4 の実演の手順を説明し、それぞれの時刻を示してください。",
        (_VIDEO,), VIDEO, id="video-content",
    ),
    pytest.param(
        "「dune_notes.pdf」 の制約を説明し、根拠の箇所を示してください。",
        (_DOCUMENT,), DOCUMENT, id="quoted-document-filename",
    ),
    pytest.param(
        "『demo.mp4』 の実演の手順と時刻を示してください。",
        (_VIDEO,), VIDEO, id="quoted-video-filename",
    ),
    pytest.param(
        "dune_notes.pdf にある「dune_notes.pdf は読まずに」という注意書きの意味を説明し、根拠の箇所を示してください。",
        (_DOCUMENT,), DOCUMENT, id="denial-is-quoted-document-content",
    ),
    pytest.param(
        "dune_notes.pdf の『demo.mp4 は開かないで』という注意書きの意味を説明し、根拠の箇所を示してください。",
        (_DOCUMENT, _VIDEO), DOCUMENT, id="quoted-denial-names-other-kind",
    ),
    pytest.param(
        "dune_notes.pdf は読まずに、other_brief.pdf の制約を説明し、根拠の箇所を示してください。",
        (_DOCUMENT, _OTHER_DOCUMENT), DOCUMENT, id="deny-one-analyze-other-document",
    ),
    pytest.param(
        "dune_notes.pdf は参照せずに、demo.mp4 の実演の手順を説明し、時刻を示してください。",
        (_DOCUMENT, _VIDEO), VIDEO, id="deny-document-analyze-video",
    ),
    pytest.param(
        "dune_notes.pdf は全文を読まずに、要点を説明し、根拠の箇所を示してください。",
        (_DOCUMENT,), DOCUMENT, id="partial-reading-is-permitted",
    ),
    pytest.param(
        "私が dune_notes.pdf を読まずに済むように要約してください。",
        (_DOCUMENT,), DOCUMENT, id="summary-spares-user-reading",
    ),
    pytest.param(
        "私が dune_notes.pdf を開かずに済むように内容を説明してください。",
        (_DOCUMENT,), DOCUMENT, id="explanation-spares-user-opening",
    ),
    pytest.param(
        "dune_notes.pdf の中で、未解決の問題を説明し、根拠の箇所を示してください。",
        (_DOCUMENT,), DOCUMENT, id="ordinary-location-comma",
    ),
]

for _boundary in ["。", "！", "？"]:
    for _filename, _evidence, _expected in [
        ("other_brief.pdf", _OTHER_DOCUMENT, DOCUMENT),
        ("demo.mp4", _VIDEO, VIDEO),
    ]:
        _POSITIVE_NEIGHBORS.append(
            pytest.param(
                "dune_notes.pdf は開かずに、"
                f"「会議は延期です」という文を丁寧に直してくれますか{_boundary}"
                f"続けて {_filename} の要点を説明し、根拠の箇所を示してください。",
                (_DOCUMENT, _evidence),
                _expected,
                id=f"rewrite-plus-analysis-{_boundary}-{_evidence.kind}",
            )
        )


@pytest.mark.parametrize("backend", ["router", "hybrid"])
@pytest.mark.parametrize(("prompt", "files", "expected"), _POSITIVE_NEIGHBORS)
def test_real_japanese_content_requests_remain_eligible_for_mock_router(
    prompt, files, expected, backend
):
    case = _case(prompt, files=files, expected=expected)
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is False
    assert probe.eligible_skills(case) == {expected}

    report, requests = _evaluate_with_mock(case, backend=backend, selected=expected)
    assert len(requests) == 1
    sample = report["backends"][backend]["samples"][0]
    assert sample["valid"] is True
    assert sample["selected"] == expected
    assert sample["source"] == "router"
    assert sample["model_called"] is True
    assert sample["terminal_abstention"] is False
    assert report["selection_check"]["status"] == "matched_labels"


@pytest.mark.parametrize("filename", ["notes。pdf", "why？.pdf", "alert！.pdf"])
def test_japanese_sentence_marks_inside_exact_quoted_filename_do_not_split_target(filename):
    case = _case(
        f"Summarize 「{filename}」.",
        files=(probe.FileEvidence(filename, "document", "ready"),),
        expected=DOCUMENT,
    )
    assert probe.eligible_skills(case) == {DOCUMENT}
    decision = probe.select_rules(case)
    assert decision.selected == DOCUMENT
    assert decision.terminal_abstention is False
