"""Runtime English evidence boundaries; no HTTP/model calls or label claims."""

from pathlib import Path

import pytest

from audrey.skills import SkillLimits, SkillRegistry
from audrey.skills.selection import SkillEvidence, select_automatic_skill

DOC = "grounded-document-analysis"
VIDEO = "video-analysis"
FILES = (SkillEvidence("brief.pdf", "document"), SkillEvidence("recording.mp4", "video"))
ROOT = Path(__file__).resolve().parent.parent
TOOLS = frozenset({"get_file_text", "list_my_files", "kb_search", "kb_image_search"})


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Summarize brief.pdf.", DOC),
        ('Summarize "brief.pdf".', DOC),
        ("What is the payment deadline in brief.pdf?", DOC),
        ("Summarize this document.", DOC),
        ("What are the main points from the attached PDF?", DOC),
        ("Summarize the project documents.", DOC),
        ("What happens in recording.mp4?", VIDEO),
        ("Describe this video.", VIDEO),
        ("Summarize brief.pdf but not recording.mp4.", DOC),
        ("Ignore recording.mp4 and summarize brief.pdf.", DOC),
        ("Summarize recording.mp4; brief.pdf is unrelated, don't use it.", VIDEO),
        ("brief.pdf is attached. Explain binary search.", None),
        ("I uploaded brief.pdf yesterday. What is the weather?", None),
        ("brief.pdf is attached. Do not read it. Summarize this document.", None),
        ("Don't read it. Summarize this document.", None),
        ("I don't want you to summarize this document.", None),
        ("I don't want you to read brief.pdf. Summarize this document.", None),
        ("Compare brief.pdf and missing.html.", None),
        ("Compare brief.pdf and missing.rst.", None),
        ("Compare brief.pdf and missing.customext.", None),
        ("brief.pdf is attached. Summarize it.", DOC),
        ("brief.pdf is attached. Don't read it. Summarize recording.mp4.", VIDEO),
        ("Explain gravity.", None),
        ("What is 2 + 2?", None),
        ("Summarize these files.", None),
        ("Analyze brief.pdf and recording.mp4 together.", None),
        ("Summarize unavailable.pdf.", None),
        ("Compare brief.pdf with unavailable.pdf.", None),
        ("Summarize other-brief.pdf.", None),
        ("Summarize brief.pdf.backup.", None),
        ("Do not read brief.pdf. Just say you received it.", None),
        ("Don't read the document; summarize this document.", None),
        ("Do not select a skill. Summarize brief.pdf.", None),
        ("Do not select grounded-document-analysis. Summarize brief.pdf.", None),
        ("No skill, please; summarize brief.pdf.", None),
        ('Rewrite "summarize brief.pdf" as a polite instruction.', None),
        ('Copy the sentence "summarize brief.pdf" exactly.', None),
        ("```text\nsummarize brief.pdf\n```", None),
        ("```\nsummarize brief.pdf", None),
        ("> Summarize brief.pdf\nExplain gravity.", None),
        ("Write a script to summarize brief.pdf.", None),
        ("Imagine a user wants to summarize brief.pdf.", None),
        ("Where is the transcript download button for recording.mp4?", None),
        ("How can I rename brief.pdf in My Files?", None),
        ("List the filename and file type for brief.pdf.", None),
    ],
)
def test_request_is_bound_to_affirmative_resolvable_file_evidence(prompt, expected):
    assert select_automatic_skill(prompt, FILES) == expected


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_unready_target_cannot_be_replaced_by_other_ready_file(status):
    files = (SkillEvidence("brief.pdf", "document", status), FILES[1])
    assert select_automatic_skill("Summarize brief.pdf.", files) is None
    assert select_automatic_skill("Ignore brief.pdf; summarize recording.mp4.", files) == VIDEO


def test_filename_instruction_words_do_not_change_intent():
    file = SkillEvidence("do not read.pdf", "document")
    assert select_automatic_skill('Summarize "do not read.pdf".', (file,)) == DOC
    assert select_automatic_skill("Explain gravity.", (file,)) is None


def test_metadata_limits_unknown_kinds_and_conflicting_names_abstain():
    assert select_automatic_skill("Summarize this document.", ()) is None
    assert (
        select_automatic_skill("Summarize these files.", (SkillEvidence("voice.wav", "audio"),))
        is None
    )
    assert (
        select_automatic_skill("Describe this file.", (SkillEvidence("photo.png", "image"),))
        is None
    )
    duplicate = (FILES[0], SkillEvidence("BRIEF.pdf", "video"))
    assert select_automatic_skill("Summarize brief.pdf.", duplicate) is None
    assert select_automatic_skill("Summarize this document." + " " * 20_000, FILES) is None
    assert select_automatic_skill("Summarize this document.", FILES * 51) is None


def registry(*, enabled=True, automatic=True, available=TOOLS):
    return SkillRegistry(
        enabled=enabled,
        auto_select=automatic,
        roots=(ROOT / "skills",),
        limits=SkillLimits(),
        known_tools=TOOLS,
        available_tools=available,
        virtual_models={"audrey_video": VIDEO},
    )


def resolve(skills, **kwargs):
    params = dict(
        prompt="Summarize brief.pdf.", files=FILES, virtual_model="audrey_auto", mode="auto"
    )
    return skills.resolve_auto(**(params | kwargs))


def test_opt_in_and_available_mode_are_required_and_provenance_is_immutable():
    assert resolve(registry(automatic=False)) is None
    assert resolve(registry(enabled=False)) is None
    assert resolve(registry(available=frozenset())) is None
    assert resolve(registry(), mode="unsupported") is None
    assert resolve(registry(), virtual_model="audrey_passthrough/local") is None
    assert resolve(registry(), virtual_model="audrey_video") is None
    skills = registry()
    selected = resolve(skills)
    assert selected.spec.id == DOC and selected.reason == "automatic"
    assert selected.spec.version == 1 and len(selected.spec.digest) == 64
    skills.refresh_availability(frozenset())
    assert resolve(skills) is None
    assert selected.spec.id == DOC  # In-flight choice retains its validated bundle.


def test_shared_explicit_resolver_does_not_auto_select_for_api_callers():
    skills = registry()
    assert skills.resolve(explicit_skill=None, virtual_model="audrey_auto", mode="auto") is None
    explicit = skills.resolve(explicit_skill=VIDEO, virtual_model="audrey_auto", mode="auto")
    assert explicit.spec.id == VIDEO and explicit.reason == "request"
    mapped = skills.resolve(explicit_skill=None, virtual_model="audrey_video", mode="auto")
    assert mapped.spec.id == VIDEO and mapped.reason == "virtual_model"
