"""The 3C document pilot is packaged and evaluated as a paired arm."""

from __future__ import annotations

import json
from pathlib import Path

from audrey.skills import SkillLimits, SkillRegistry, load_skill_bundle
from audrey.tools.discovery import TOOL_DECLARATIONS

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
CASES = ROOT / "scripts" / "eval_prompts_grounded_documents.json"
FIXTURES = ROOT / "scripts" / "fixtures" / "grounded-document-analysis"
KNOWN_TOOLS = frozenset(TOOL_DECLARATIONS)
DOCUMENT_TOOLS = frozenset({
    "list_my_files",
    "get_file_text",
    "kb_search",
    "kb_image_search",
})


def test_tracked_grounded_document_bundle_has_the_bounded_contract():
    spec = load_skill_bundle(
        SKILLS / "grounded-document-analysis",
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
    )

    assert spec.id == "grounded-document-analysis"
    assert spec.name == "Grounded document analysis"
    assert spec.version == 1
    assert spec.allowed_tools == DOCUMENT_TOOLS
    assert spec.supported_modes == frozenset({"auto", "fast", "deep"})
    assert spec.resources == ()
    assert "one entry per file" in spec.instructions
    assert "unread or unavailable portion" in spec.instructions
    assert "content from another file cannot stand in for it" in spec.instructions


def test_repository_registry_loads_both_skills_without_changing_video_mapping():
    registry = SkillRegistry(
        enabled=True,
        roots=(SKILLS,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
        virtual_models={"audrey_video": "video-analysis"},
    )

    snapshot = registry.snapshot()
    assert snapshot.status == "ready"
    assert snapshot.loaded_count == 2
    assert snapshot.available_count == 2
    assert [item.id for item in registry.catalog()] == [
        "grounded-document-analysis",
        "video-analysis",
    ]

    video = registry.resolve_virtual_model("audrey_video")
    document = registry.resolve(
        explicit_skill="grounded-document-analysis",
        virtual_model="audrey_auto",
        mode="auto",
    )
    assert video is not None and video.spec.id == "video-analysis"
    assert document is not None
    assert document.spec.id == "grounded-document-analysis"
    assert document.reason == "request"


def test_grounded_document_eval_interleaves_identical_control_and_skill_arms():
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    assert len(cases) == 6
    for control, skill in zip(cases[::2], cases[1::2], strict=True):
        assert control["name"].endswith("-control")
        assert skill["name"] == control["name"].removesuffix("-control") + "-skill"
        assert "skill" not in control
        assert skill["skill"] == "grounded-document-analysis"

        control_contract = {k: v for k, v in control.items() if k != "name"}
        skill_contract = {k: v for k, v in skill.items() if k not in {"name", "skill"}}
        assert skill_contract == control_contract


def test_grounded_document_eval_fixtures_pin_the_objective_facts():
    operations = (FIXTURES / "c3-grounded-operations.md").read_text(encoding="utf-8")
    support = (FIXTURES / "c3-grounded-support.md").read_text(encoding="utf-8")
    cases = CASES.read_text(encoding="utf-8")

    assert all(value in operations for value in ("Alder", "October 14, 2026", "840 ms"))
    assert all(value in support for value in ("18 minutes", "34 tickets per hour"))
    assert "customer satisfaction" in operations
    assert "customer satisfaction" in support
    assert "c3-grounded-operations.md" in cases
    assert "c3-grounded-support.md" in cases
