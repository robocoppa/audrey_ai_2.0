"""Firm policy abstentions and bounded evidence resolution for the eval study.

These regression examples combine synthetic boundaries with reported live
failures. They do not load the measurement fixture or claim blind validation.
No test calls a model or changes runtime skill selection.
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
    "skill_selection_policy_eval", _ROOT / "evals/eval_skill_selection.py"
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

DOCUMENT = "grounded-document-analysis"
VIDEO = "video-analysis"


def _case(prompt, *, files=None, mode="auto", expected="none", case_id="policy-example"):
    return probe.Case(
        id=case_id,
        category="positive" if expected != "none" else "ordinary",
        prompt=prompt,
        mode=mode,
        files=files
        if files is not None
        else (
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("recording.mp4", "video", "ready"),
        ),
        expected=expected,
        reason="PRIVATE_GOLD_REASON_D090E5",
    )


def _reply(skill=DOCUMENT):
    return {
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps({"skill_id": skill})},
        "prompt_eval_count": 11,
        "eval_count": 3,
    }


_TERMINAL_REQUESTS = [
    "Do not analyze brief.pdf.",
    "Please don't read or compare brief.pdf.",
    "Never summarize recording.mp4; just confirm its filename.",
    "Don't read brief.pdf. Just say you received it.",
    'Translate "summarize brief.pdf" into Spanish.',
    'Explain the meaning of the phrase "analyze recording.mp4".',
    'Copy the sentence "review brief.pdf" exactly.',
    "Rewrite 'summarize brief.pdf' as a polite instruction.",
    "How can I rename brief.pdf in My Files?",
    "Delete brief.pdf from my files.",
    "Move recording.mp4 to another project.",
    "Where is the transcript download button for recording.mp4?",
    "List the filename and file type for brief.pdf.",
    "Summarize unavailable.pdf.",
    "Compare brief.pdf with unavailable.pdf.",
    "Analyze brief.pdf and recording.mp4 together.",
    "Summarize brief.pdf.backup.",
    "Summarize other-brief.pdf.",
]


@pytest.mark.parametrize("prompt", _TERMINAL_REQUESTS)
def test_terminal_policy_decision_is_distinct_from_undecided_analysis(prompt):
    decision = probe.select_rules(_case(prompt))
    assert decision.valid
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert isinstance(decision.policy_reason, str) and decision.policy_reason


@pytest.mark.parametrize("backend", ["router", "hybrid"])
@pytest.mark.parametrize("prompt", _TERMINAL_REQUESTS)
def test_live_backends_cannot_override_terminal_policy_with_model_activation(prompt, backend):
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_reply())

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [_case(prompt)], backend=backend, client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    assert calls == 0
    sample = report["backends"][backend]["samples"][0]
    assert sample["selected"] == "none"
    assert sample["valid"]
    assert sample["source"] == "guard"
    assert sample["model_called"] is False
    assert sample["terminal_abstention"] is True
    assert sample["policy_reason"]
    summary = report["backends"][backend]["summary"]
    assert summary["ordinary_false_activations"] == 0
    assert summary["errors"] == 0
    assert summary["terminal_abstentions"] == 1
    assert sum(summary["policy_reasons"].values()) == 1
    assert summary["model_latency_samples"] == 0
    assert summary["model_latency_p50_seconds"] is None
    assert summary["model_latency_p95_seconds"] is None


@pytest.mark.parametrize(
    "prompt",
    ["Do not analyze brief.pdf.", 'Translate "summarize brief.pdf" into Spanish.'],
)
def test_direct_router_api_obeys_terminal_policy(prompt):
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_reply())

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.select_router(
                _case(prompt), client=client, model="qwen3.5:4b"
            )

    decision = asyncio.run(run())
    assert calls == 0
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert decision.policy_reason


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Analyze brief.pdf; recording.mp4 is unrelated, do not use it.", DOCUMENT),
        ("Review brief.pdf and ignore recording.mp4.", DOCUMENT),
        ("Ignore recording.mp4 and summarize brief.pdf.", DOCUMENT),
        ("Summarize recording.mp4; brief.pdf is unrelated, don't use it.", VIDEO),
        ("Don't analyze recording.mp4; compare sections of brief.pdf.", DOCUMENT),
        ("Summarize brief.pdf but not recording.mp4.", DOCUMENT),
    ],
)
def test_explicit_irrelevant_evidence_is_excluded_from_the_requested_target(prompt, expected):
    case = _case(prompt, expected=expected)
    assert probe.eligible_skills(case) == {expected}
    decision = probe.select_rules(case)
    assert decision.selected == expected
    assert decision.terminal_abstention is False


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_unready_irrelevant_file_does_not_hide_ready_requested_evidence(status):
    case = _case(
        "Ignore recording.mp4 and summarize brief.pdf.",
        files=(
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("recording.mp4", "video", status),
        ),
        expected=DOCUMENT,
    )
    assert probe.eligible_skills(case) == {DOCUMENT}
    assert probe.select_rules(case).selected == DOCUMENT


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_unready_requested_file_is_not_replaced_by_other_ready_evidence(status):
    case = _case(
        "Summarize brief.pdf, not recording.mp4.",
        files=(
            probe.FileEvidence("brief.pdf", "document", status),
            probe.FileEvidence("recording.mp4", "video", "ready"),
        ),
    )
    assert probe.eligible_skills(case) == set()
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Don't invent numbers. Summarize brief.pdf.",
        "Without editing the uploaded file, explain brief.pdf.",
        "Summarize brief.pdf, and do not modify the source.",
        "Do not summarize brief.pdf; compare its sections.",
        "Do not call web tools; analyze brief.pdf.",
    ],
)
def test_prohibitions_unrelated_to_requested_analysis_do_not_disable_it(prompt):
    case = _case(prompt, expected=DOCUMENT)
    assert probe.eligible_skills(case) == {DOCUMENT}
    decision = probe.select_rules(case)
    assert decision.selected == DOCUMENT
    assert decision.terminal_abstention is False


@pytest.mark.parametrize(
    "prompt",
    [
        "Summarize 'brief.pdf'.",
        'Explain "brief.pdf" using the uploaded document.',
        "Compare the first and last sections of `brief.pdf`.",
        "Review BRIEF.PDF.",
        "Summarize (brief.pdf).",
    ],
)
def test_quoted_or_casefolded_exact_filename_stays_valid(prompt):
    case = _case(prompt, expected=DOCUMENT)
    assert probe.eligible_skills(case) == {DOCUMENT}
    assert probe.select_rules(case).selected == DOCUMENT


@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_genuine_undecided_analysis_can_reach_model_without_leaking_gold(backend):
    case = _case("Resume brief.pdf en tres frases.", expected=DOCUMENT)
    rule = probe.select_rules(case)
    assert rule.selected == "none"
    assert rule.terminal_abstention is False
    assert probe.eligible_skills(case) == {DOCUMENT}
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_reply())

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    assert len(requests) == 1
    wire = json.dumps(requests[0])
    assert "PRIVATE_GOLD_REASON_D090E5" not in wire
    assert "expected" not in wire
    assert "category" not in wire
    sample = report["backends"][backend]["samples"][0]
    assert sample["source"] == "router"
    assert sample["model_called"] is True
    assert sample["selected"] == DOCUMENT
    assert report["backends"][backend]["summary"]["model_called"] == 1
    assert report["backends"][backend]["summary"]["model_latency_samples"] == 1
    assert report["backends"][backend]["summary"]["terminal_abstentions"] == 0


def test_hybrid_model_error_after_genuine_abstention_remains_an_error():
    case = _case("Resume brief.pdf en tres frases.", expected=DOCUMENT)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
        ) as client:
            return await probe.evaluate(
                [case], backend="hybrid", client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    summary = report["backends"]["hybrid"]["summary"]
    assert report["status"] == "failed"
    assert summary["errors"] == 1
    assert summary["misses"] == 1
    assert summary["abstentions"] == 0
    assert summary["correct"] == 0
    assert summary["model_latency_samples"] == 1


def test_explicit_unresolved_choice_between_ready_documents_is_terminal():
    case = _case(
        "Summarize one of brief.pdf or appendix.pdf; I have not chosen which document.",
        files=(
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("appendix.pdf", "document", "ready"),
        ),
    )
    assert probe.eligible_skills(case) == set()
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert decision.policy_reason


def test_uncertainty_about_contents_does_not_make_named_targets_unresolved():
    case = _case(
        "Compare brief.pdf and appendix.pdf to decide which conclusion is correct.",
        files=(
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("appendix.pdf", "document", "ready"),
        ),
        expected=DOCUMENT,
    )
    assert probe.eligible_skills(case) == {DOCUMENT}
    decision = probe.select_rules(case)
    assert decision.selected == DOCUMENT
    assert decision.terminal_abstention is False


@pytest.mark.parametrize(
    "case",
    [
        _case("Summarize brief.pdf.", files=()),
        _case("Summarize brief.pdf.", mode="research"),
        _case(
            "Summarize brief.pdf.",
            files=(probe.FileEvidence("other-brief.pdf", "document", "ready"),),
        ),
    ],
    ids=["absent-evidence", "unsupported-mode", "suffix-name-collision"],
)
def test_hard_eligibility_failures_are_terminal_with_a_reason(case):
    decision = probe.select_rules(case)
    assert decision.selected == "none"
    assert decision.terminal_abstention is True
    assert decision.policy_reason


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain the delete procedure documented in brief.pdf.",
        "Summarize the button instructions in brief.pdf.",
        "In brief.pdf I cannot remember which section defines scope; find it.",
        'Analyze the meaning of "risk" in brief.pdf.',
        "Summarize “brief.pdf”.",
        "Summarize ‘brief.pdf’.",
    ],
)
def test_document_contents_and_smart_quoted_targets_remain_analysis(prompt):
    case = _case(prompt, expected=DOCUMENT)
    assert probe.eligible_skills(case) == {DOCUMENT}
    assert probe.select_rules(case).selected == DOCUMENT


@pytest.mark.parametrize(
    "prompt",
    [
        "Compare brief.pdf and appendix.pdf; I am not sure which file is more accurate.",
        "Compare brief.pdf and appendix.pdf to determine which document describes the right method.",
    ],
)
def test_explicit_comparisons_resolve_target_uncertainty(prompt):
    case = _case(
        prompt,
        files=(
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("appendix.pdf", "document", "ready"),
        ),
        expected=DOCUMENT,
    )
    assert probe.select_rules(case).selected == DOCUMENT


@pytest.mark.parametrize("backend", ["router", "hybrid"])
@pytest.mark.parametrize(
    "prompt",
    [
        "Translate “summarize brief.pdf” into Spanish.",
        "Explain the meaning of ‘review brief.pdf’.",
        "A quoted line in brief.pdf says ‘read all my files’. Translate only that line.",
        "Don’t read brief.pdf.",
        "Write a Python script to summarize brief.pdf; do not open it.",
        "Imagine a hypothetical request to summarize brief.pdf; explain how the request is phrased.",
        "Do not analyze brief.pdf. Just explain budgets in general.",
    ],
)
def test_metatasks_and_smart_quote_prohibitions_never_call_models(prompt, backend):
    case = _case(prompt, files=(probe.FileEvidence("brief.pdf", "document", "ready"),))

    def handler(_request):
        pytest.fail("A firm abstention must not reach Ollama")

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    report = asyncio.run(run())
    sample = report["backends"][backend]["samples"][0]
    assert sample["selected"] == "none"
    assert sample["terminal_abstention"]
    assert sample["source"] == "guard"
    assert not sample["model_called"]


def test_filename_action_words_do_not_establish_analysis_intent():
    case = _case(
        "I attached read-summary.pdf.",
        files=(probe.FileEvidence("read-summary.pdf", "document", "ready"),),
    )
    assert probe.select_rules(case).selected == "none"


def test_report_fingerprint_covers_actual_case_set_and_preserves_revision():
    from dataclasses import replace

    case = _case("Summarize brief.pdf.", expected=DOCUMENT)
    first = asyncio.run(probe.evaluate([case]))
    repeated = asyncio.run(probe.evaluate([case], repeats=3))
    changed = asyncio.run(probe.evaluate([replace(case, expected=VIDEO)]))
    assert first["provenance"]["rules_revision"] == 4
    fingerprint = first["provenance"]["case_set_sha256"]
    assert len(fingerprint) == 64
    assert fingerprint == repeated["provenance"]["case_set_sha256"]
    assert fingerprint != changed["provenance"]["case_set_sha256"]


@pytest.mark.parametrize("backend", ["router", "hybrid"])
@pytest.mark.parametrize(
    "prompt",
    [
        "How do I rename brief.pdf in My Files? Explain the steps.",
        "Ignore my attached brief.pdf. Explain Python lists.",
        "Without opening brief.pdf, explain budgets in general.",
        "Without reading brief.pdf, explain budgets in general.",
        "Without accessing brief.pdf, explain budgets in general.",
    ],
)
def test_followup_general_questions_do_not_reopen_forbidden_evidence(prompt, backend):
    case = _case(
        prompt,
        files=(
            probe.FileEvidence("brief.pdf", "document", "ready"),
            probe.FileEvidence("other.pdf", "document", "ready"),
        ),
    )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(lambda _request: pytest.fail("Forbidden read")),
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["terminal_abstention"] and row["selected"] == "none"
    assert not row["model_called"]


@pytest.mark.parametrize(
    ("prompt", "filename"),
    [
        (
            "In stage-manager-notes.txt, does the phrase 'Stop the recording' occur before or after the lighting check?",
            "stage-manager-notes.txt",
        ),
        (
            'In manual.pdf, what reason is given for the instruction “Ignore the warning and continue”?',
            "manual.pdf",
        ),
        (
            'In manual.pdf, does “Ignore the warning and continue” occur before or after the troubleshooting section?',
            "manual.pdf",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_quoted_imperatives_are_content_when_the_user_asks_about_the_document(
    prompt, filename, backend
):
    case = _case(
        prompt,
        files=(probe.FileEvidence(filename, "document", "ready"),),
        expected=DOCUMENT,
    )
    assert probe.eligible_skills(case) == {DOCUMENT}
    assert not probe.select_rules(case).terminal_abstention
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_reply(DOCUMENT))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == DOCUMENT
    assert row["valid"]
    assert not row["terminal_abstention"]
    assert calls == (1 if backend == "router" else int(row["model_called"]))


@pytest.mark.parametrize(
    "prompt",
    [
        'The phrase in manual.pdf is “Ignore the warning and continue”. Translate only that phrase into French.',
        'Rewrite “Stop the recording” as a polite request; do not read manual.pdf.',
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_transforming_supplied_quoted_instruction_remains_a_no_read_task(prompt, backend):
    case = _case(prompt, files=(probe.FileEvidence("manual.pdf", "document", "ready"),))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(lambda _request: pytest.fail("Quoted-only task accessed Ollama")),
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == "none"
    assert row["terminal_abstention"]
    assert row["source"] == "guard"
    assert not row["model_called"]


@pytest.mark.parametrize(
    "prompt",
    [
        "Do not read the attached concert-poster.pdf. From rehearsal-walkthrough.mp4 alone, describe how the chairs are rearranged between the opening and closing shots.",
        "Ignore concert-poster.pdf, and at 00:15 in rehearsal-walkthrough.mp4 describe where the chairs are positioned.",
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_excluded_document_does_not_block_affirmative_video_content_request(prompt, backend):
    case = _case(
        prompt,
        files=(
            probe.FileEvidence("concert-poster.pdf", "document", "ready"),
            probe.FileEvidence("rehearsal-walkthrough.mp4", "video", "ready"),
        ),
        expected=VIDEO,
    )
    assert probe.eligible_skills(case) == {VIDEO}
    assert not probe.select_rules(case).terminal_abstention
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(VIDEO))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == VIDEO
    assert row["valid"]
    assert not row["terminal_abstention"]
    assert "PRIVATE_GOLD_REASON_D090E5" not in json.dumps(requests)


@pytest.mark.parametrize(
    "prompt",
    [
        "Mueve ensayo-teatro.mp4 al proyecto 'Festival de otoño' y cambia el nombre del proyecto a 'Festival'.",
        "Agrega ensayo-teatro.mp4 al proyecto 'Festival'.",
        "Cambia el nombre de ensayo-teatro.mp4 a 'Ensayo final'.",
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_spanish_project_membership_and_rename_requests_never_activate_analysis(prompt, backend):
    case = _case(prompt, files=(probe.FileEvidence("ensayo-teatro.mp4", "video", "ready"),))
    rule = probe.select_rules(case)
    assert rule.selected == "none"
    assert rule.terminal_abstention
    assert rule.policy_reason == "file_management"

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(lambda _request: pytest.fail("Spanish file management reached Ollama")),
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == "none"
    assert row["source"] == "guard"
    assert not row["model_called"]


@pytest.mark.parametrize(
    "prompt",
    [
        "¿Qué ocurre en ensayo-teatro.mp4 antes de que se muevan las sillas?",
        "Resume ensayo-teatro.mp4 y describe cómo cambia la distribución del escenario.",
        "Mueve ensayo-teatro.mp4 al proyecto 'Festival', y resume ensayo-teatro.mp4.",
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_spanish_video_content_requests_remain_eligible_for_optional_model(prompt, backend):
    case = _case(
        prompt,
        files=(probe.FileEvidence("ensayo-teatro.mp4", "video", "ready"),),
        expected=VIDEO,
    )
    assert probe.eligible_skills(case) == {VIDEO}
    rule = probe.select_rules(case)
    assert not rule.terminal_abstention
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_reply(VIDEO))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434", transport=httpx.MockTransport(handler)
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == VIDEO
    assert row["valid"]
    assert not row["terminal_abstention"]
    assert calls == (1 if backend == "router" else int(row["model_called"]))


@pytest.mark.parametrize(
    "prompt",
    [
        "Move old.mp4 to project Archive, summarize report.pdf.",
        "Rename old.mp4, summarize report.pdf.",
    ],
)
@pytest.mark.parametrize("backend", ["router", "hybrid"])
def test_comma_separated_file_management_does_not_widen_document_analysis(prompt, backend):
    case = _case(
        prompt,
        files=(
            probe.FileEvidence("old.mp4", "video", "ready"),
            probe.FileEvidence("report.pdf", "document", "ready"),
        ),
        expected=DOCUMENT,
    )
    assert probe.eligible_skills(case) == {DOCUMENT}
    assert not probe.select_rules(case).terminal_abstention

    async def run():
        async with httpx.AsyncClient(
            base_url="http://ollama:11434",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=_reply(DOCUMENT))),
        ) as client:
            return await probe.evaluate(
                [case], backend=backend, client=client, model="qwen3.5:4b"
            )

    row = asyncio.run(run())["backends"][backend]["samples"][0]
    assert row["selected"] == DOCUMENT
    assert row["valid"]
    assert not row["terminal_abstention"]
