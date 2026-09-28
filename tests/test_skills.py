"""Skill bundles are strict, inert, bounded, and atomically discoverable."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from audrey.pipeline.prompts import VIDEO_SPECIALIST_SYSTEM
from audrey.skills import SkillLimits, SkillLoadError, SkillRegistry, load_skill_bundle
from audrey.tools.discovery import TOOL_DECLARATIONS

KNOWN_TOOLS = frozenset({"kb_search", "list_my_files"})
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_bundle(
    root: Path,
    skill_id: str,
    *,
    version: int = 1,
    allowed_tools: list[str] | None = None,
    resources: dict[str, str] | None = None,
    instructions: str = "Follow the bounded workflow.\n",
    extra: dict | None = None,
) -> Path:
    bundle = root / skill_id
    bundle.mkdir(parents=True)
    resource_values = resources or {}
    manifest = {
        "id": skill_id,
        "name": "Video analysis",
        "description": "Analyze uploaded video evidence.",
        "version": version,
        "allowed_tools": allowed_tools or [],
        "supported_modes": ["auto", "fast", "deep"],
        "resources": list(resource_values),
        **(extra or {}),
    }
    front_matter = yaml.safe_dump(manifest, sort_keys=False)
    (bundle / "SKILL.md").write_text(
        f"---\n{front_matter}---\n{instructions}",
        encoding="utf-8",
    )
    for relative, content in resource_values.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return bundle


def _load(bundle: Path, *, limits: SkillLimits | None = None):
    return load_skill_bundle(
        bundle,
        limits=limits or SkillLimits(),
        known_tools=KNOWN_TOOLS,
    )


def test_valid_bundle_preserves_instructions_and_hashes_resources(tmp_path):
    bundle = _write_bundle(
        tmp_path,
        "video-analysis",
        allowed_tools=["kb_search"],
        resources={"references/rules.md": "Ground every conclusion.\n"},
        instructions="\nUse the evidence exactly.\n",
    )

    first = _load(bundle)
    second = _load(bundle)

    assert first.id == "video-analysis"
    assert first.instructions == "\nUse the evidence exactly.\n"
    assert first.allowed_tools == frozenset({"kb_search"})
    assert first.resources[0].path == "references/rules.md"
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_tracked_video_bundle_matches_rollback_prompt_byte_for_byte():
    spec = load_skill_bundle(
        _REPO_ROOT / "skills" / "video-analysis",
        limits=SkillLimits(),
        known_tools=frozenset(TOOL_DECLARATIONS),
    )

    assert spec.id == "video-analysis"
    assert spec.instructions == VIDEO_SPECIALIST_SYSTEM
    assert spec.resources == ()


@pytest.mark.parametrize(
    ("skill_text", "code"),
    [
        ("id: duplicate\nid: duplicate\n", "duplicate_manifest_key"),
        ("id: [unterminated\n", "invalid_yaml"),
        ("id: &value duplicate\nname: *value\n", "yaml_features_not_allowed"),
    ],
)
def test_manifest_rejects_ambiguous_yaml(tmp_path, skill_text, code):
    bundle = tmp_path / "duplicate"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        f"---\n{skill_text}---\ninstructions\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillLoadError, match=code) as exc:
        _load(bundle)

    assert exc.value.code == code


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"surprise": True}, "unknown_manifest_key"),
        ({"id": "different-id"}, "invalid_id"),
        ({"version": 0}, "invalid_version"),
        ({"allowed_tools": ["not_declared"]}, "unknown_tool"),
        ({"supported_modes": ["research"]}, "invalid_supported_modes"),
        ({"resources": ["../secret.md"]}, "invalid_resource_path"),
        ({"resources": ["references/./rules.md"]}, "invalid_resource_path"),
        ({"resources": ["references//rules.md"]}, "invalid_resource_path"),
        ({"resources": ["references/data.txt"]}, "invalid_resource_path"),
    ],
)
def test_manifest_contract_fails_closed(tmp_path, mutation, code):
    bundle = _write_bundle(tmp_path, "video-analysis", extra=mutation)

    with pytest.raises(SkillLoadError, match=code):
        _load(bundle)


def test_loader_rejects_symlinks_executables_and_undeclared_files(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")

    symlink_bundle = _write_bundle(
        tmp_path / "symlink-root",
        "video-analysis",
        resources={"references/rules.md": "placeholder"},
    )
    resource = symlink_bundle / "references/rules.md"
    resource.unlink()
    resource.symlink_to(outside)
    with pytest.raises(SkillLoadError, match="symlink_not_allowed"):
        _load(symlink_bundle)

    executable_bundle = _write_bundle(tmp_path / "exec-root", "video-analysis")
    skill_file = executable_bundle / "SKILL.md"
    skill_file.chmod(0o755)
    with pytest.raises(SkillLoadError, match="executable_not_allowed"):
        _load(executable_bundle)

    extra_bundle = _write_bundle(tmp_path / "extra-root", "video-analysis")
    (extra_bundle / "notes.md").write_text("undeclared", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="undeclared_bundle_entry"):
        _load(extra_bundle)


def test_unreadable_resource_directory_degrades_as_a_bundle_error(
    monkeypatch,
    tmp_path,
):
    bundle = _write_bundle(
        tmp_path,
        "video-analysis",
        resources={"references/rules.md": "rules"},
    )
    original = Path.iterdir

    def _iterdir(path):
        if path == bundle / "references":
            raise PermissionError("private path")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", _iterdir)

    with pytest.raises(SkillLoadError, match="unreadable_bundle"):
        _load(bundle)


def test_loader_enforces_instruction_resource_and_total_caps(tmp_path):
    instruction_bundle = _write_bundle(
        tmp_path / "instruction-root",
        "video-analysis",
        instructions="123456",
    )
    with pytest.raises(SkillLoadError, match="instructions_too_large"):
        _load(
            instruction_bundle,
            limits=SkillLimits(
                max_instruction_chars=5,
                max_resource_chars=1000,
                max_bundle_chars=1000,
            ),
        )

    resource_bundle = _write_bundle(
        tmp_path / "resource-root",
        "video-analysis",
        resources={"references/rules.md": "123456"},
    )
    with pytest.raises(SkillLoadError, match="resource_too_large"):
        _load(
            resource_bundle,
            limits=SkillLimits(
                max_instruction_chars=1000,
                max_resource_chars=5,
                max_bundle_chars=1000,
            ),
        )

    skill_chars = len((resource_bundle / "SKILL.md").read_text(encoding="utf-8"))
    with pytest.raises(SkillLoadError, match="bundle_too_large"):
        _load(
            resource_bundle,
            limits=SkillLimits(
                max_instruction_chars=1000,
                max_resource_chars=1000,
                max_bundle_chars=skill_chars + 5,
            ),
        )


def test_disabled_registry_does_not_touch_the_filesystem(monkeypatch, tmp_path):
    touched = False

    def _unexpected_lstat(_self):
        nonlocal touched
        touched = True
        raise AssertionError("disabled registry read the filesystem")

    monkeypatch.setattr(Path, "lstat", _unexpected_lstat)
    registry = SkillRegistry(
        enabled=False,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )

    assert registry.snapshot().status == "disabled"
    assert touched is False


def test_empty_deployment_root_ignores_only_its_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )

    assert registry.snapshot().status == "ready"
    assert registry.snapshot().loaded_count == 0


def test_registry_isolates_invalid_bundles_and_degrades_missing_tools(tmp_path):
    _write_bundle(
        tmp_path,
        "available-skill",
        allowed_tools=["kb_search"],
    )
    _write_bundle(
        tmp_path,
        "degraded-skill",
        allowed_tools=["list_my_files"],
    )
    _write_bundle(
        tmp_path,
        "invalid-skill",
        extra={"unknown": True},
    )

    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=frozenset({"kb_search"}),
    )
    snapshot = registry.snapshot()

    assert snapshot.status == "degraded"
    assert snapshot.loaded_count == 2
    assert snapshot.available_count == 1
    assert snapshot.degraded_count == 1
    assert snapshot.invalid_count == 1
    assert snapshot.invalid[0].code == "unknown_manifest_key"
    assert registry.get("invalid-skill") is None
    assert {item.id for item in registry.catalog()} == {
        "available-skill",
        "degraded-skill",
    }


def test_duplicate_ids_across_roots_fail_closed(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_bundle(first_root, "same-skill", version=1)
    _write_bundle(second_root, "same-skill", version=2)

    registry = SkillRegistry(
        enabled=True,
        roots=(first_root, second_root),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )

    assert registry.get("same-skill") is None
    assert registry.snapshot().status == "unavailable"
    assert registry.snapshot().invalid[0].code == "duplicate_skill_id"


def test_rediscovery_swaps_immutable_specs_and_refresh_does_not_reread(tmp_path):
    bundle = _write_bundle(tmp_path, "video-analysis", version=1)
    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )
    old_spec = registry.get("video-analysis").spec

    skill_text = (bundle / "SKILL.md").read_text(encoding="utf-8")
    (bundle / "SKILL.md").write_text(
        skill_text.replace("version: 1", "version: 2"),
        encoding="utf-8",
    )
    registry.rediscover()
    new_spec = registry.get("video-analysis").spec

    assert old_spec.version == 1
    assert new_spec.version == 2
    assert old_spec.digest != new_spec.digest

    (bundle / "SKILL.md").unlink()
    registry.refresh_availability(frozenset())
    assert registry.get("video-analysis").spec is new_spec


def test_catalog_never_exposes_instructions_resources_or_paths(tmp_path):
    _write_bundle(
        tmp_path,
        "video-analysis",
        resources={"templates/private.md": "private template"},
        instructions="private instructions",
    )
    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )

    catalog = repr(registry.catalog())

    assert "private instructions" not in catalog
    assert "private template" not in catalog
    assert "templates/private.md" not in catalog


def test_virtual_model_mapping_resolves_only_an_available_bundle(tmp_path):
    _write_bundle(
        tmp_path,
        "video-analysis",
        allowed_tools=["kb_search"],
        instructions="Use the tracked workflow.",
    )
    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=frozenset({"kb_search"}),
        virtual_models={"audrey_video": "video-analysis"},
    )

    record = registry.resolve_virtual_model("audrey_video")
    assert record is not None
    assert record.spec.instructions == "Use the tracked workflow."
    assert registry.resolve_virtual_model("audrey_auto") is None

    registry.refresh_availability(frozenset())
    assert registry.resolve_virtual_model("audrey_video") is None
    assert registry.snapshot().status == "degraded"


def test_missing_mapped_bundle_is_reported_as_unavailable(tmp_path):
    registry = SkillRegistry(
        enabled=True,
        roots=(tmp_path,),
        limits=SkillLimits(),
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
        virtual_models={"audrey_video": "video-analysis"},
    )

    snapshot = registry.snapshot()
    assert snapshot.status == "unavailable"
    assert snapshot.invalid_count == 1
    assert snapshot.invalid[0].skill_id == "video-analysis"
    assert snapshot.invalid[0].code == "mapped_skill_unavailable"


def test_explicit_selection_resolves_supported_modes_and_stays_immutable(tmp_path):
    bundle = _write_bundle(tmp_path, "video-analysis", version=1)
    registry = SkillRegistry(
        enabled=True, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=KNOWN_TOOLS, available_tools=KNOWN_TOOLS,
        virtual_models={"audrey_video": "video-analysis"},
    )

    for virtual_model, mode in (("audrey_auto", "auto"), ("audrey_fast", "fast"), ("audrey_deep", "deep")):
        resolved = registry.resolve(
            explicit_skill="video-analysis",
            virtual_model=virtual_model,
            mode=mode,
        )
        assert resolved is not None
        assert resolved.reason == "request"
        assert resolved.spec.version == 1

    held = registry.resolve(
        explicit_skill=None, virtual_model="audrey_video", mode="auto",
    )
    text = (bundle / "SKILL.md").read_text(encoding="utf-8")
    (bundle / "SKILL.md").write_text(
        text.replace("version: 1", "version: 2"), encoding="utf-8",
    )
    registry.rediscover()

    assert held is not None
    assert held.reason == "virtual_model"
    assert held.spec.version == 1
    assert registry.get("video-analysis").spec.version == 2


def test_explicit_selection_rejects_unknown_conflicting_and_unavailable(tmp_path):
    _write_bundle(tmp_path, "video-analysis", allowed_tools=["kb_search"])
    registry = SkillRegistry(
        enabled=True, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=KNOWN_TOOLS, available_tools=frozenset(),
        virtual_models={"audrey_video": "video-analysis"},
    )

    with pytest.raises(ValueError, match="Unknown skill") as unknown:
        registry.resolve(
            explicit_skill="missing-skill", virtual_model="audrey_auto", mode="auto",
        )
    assert unknown.value.code == "unknown_skill"

    with pytest.raises(ValueError, match="conflicts") as conflict:
        registry.resolve(
            explicit_skill="missing-skill", virtual_model="audrey_video", mode="auto",
        )
    assert conflict.value.code == "skill_conflict"

    with pytest.raises(ValueError, match="unavailable") as unavailable:
        registry.resolve(
            explicit_skill="video-analysis", virtual_model="audrey_auto", mode="auto",
        )
    assert unavailable.value.code == "skill_unavailable"
    assert unavailable.value.unavailable_tools == ("kb_search",)


def test_explicit_selection_rejects_disabled_and_unsupported_modes(tmp_path):
    _write_bundle(
        tmp_path, "fast-only", extra={"supported_modes": ["fast"]},
    )
    enabled = SkillRegistry(
        enabled=True, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=KNOWN_TOOLS, available_tools=KNOWN_TOOLS,
    )
    with pytest.raises(ValueError, match="does not support") as unsupported:
        enabled.resolve(
            explicit_skill="fast-only", virtual_model="audrey_deep", mode="deep",
        )
    assert unsupported.value.code == "skill_mode_unsupported"

    implicit = SkillRegistry(
        enabled=True, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=KNOWN_TOOLS, available_tools=KNOWN_TOOLS,
        virtual_models={"audrey_video": "fast-only"},
    )
    assert implicit.resolve(
        explicit_skill=None, virtual_model="audrey_video", mode="auto",
    ) is None

    disabled = SkillRegistry(
        enabled=False, roots=(tmp_path,), limits=SkillLimits(),
        known_tools=KNOWN_TOOLS, available_tools=KNOWN_TOOLS,
    )
    with pytest.raises(ValueError, match="disabled") as error:
        disabled.resolve(
            explicit_skill="fast-only", virtual_model="audrey_fast", mode="fast",
        )
    assert error.value.code == "skills_disabled"
