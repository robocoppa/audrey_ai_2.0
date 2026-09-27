"""Strict loader for declarative, non-executable Audrey skill bundles."""

from __future__ import annotations

import hashlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml
from yaml.nodes import MappingNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

from audrey.skills.models import SkillMode, SkillResource, SkillSpec

_SKILL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SUPPORTED_MODES = frozenset({"auto", "fast", "deep"})
_MANIFEST_KEYS = frozenset({
    "id",
    "name",
    "description",
    "version",
    "allowed_tools",
    "supported_modes",
    "resources",
})
_REQUIRED_KEYS = _MANIFEST_KEYS - {"resources"}


@dataclass(frozen=True, slots=True)
class SkillLimits:
    max_instruction_chars: int = 12_000
    max_resource_chars: int = 12_000
    max_bundle_chars: int = 32_000


class SkillLoadError(ValueError):
    """A bundle failed validation; ``code`` is safe for readiness output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise SkillLoadError("invalid_manifest") from exc
        if duplicate:
            raise SkillLoadError("duplicate_manifest_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _read_text_file(path: Path, *, max_chars: int, oversize_code: str) -> str:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SkillLoadError("unreadable_file") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SkillLoadError("symlink_not_allowed")
    if not stat.S_ISREG(info.st_mode):
        raise SkillLoadError("non_regular_file")
    if info.st_mode & 0o111:
        raise SkillLoadError("executable_not_allowed")
    # UTF-8 uses at most four bytes per character. This bounds the read before
    # decoding while the authoritative contract remains a character limit.
    if info.st_size > max_chars * 4:
        raise SkillLoadError(oversize_code)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SkillLoadError("invalid_utf8") from exc
    except OSError as exc:
        raise SkillLoadError("unreadable_file") from exc
    if len(text) > max_chars:
        raise SkillLoadError(oversize_code)
    return text


def _split_skill_file(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillLoadError("missing_front_matter")
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            manifest = "".join(lines[1:index])
            instructions = "".join(lines[index + 1 :])
            if not instructions.strip():
                raise SkillLoadError("empty_instructions")
            return manifest, instructions
    raise SkillLoadError("missing_front_matter_end")


def _load_manifest(text: str) -> dict[str, Any]:
    try:
        if any(isinstance(token, (AliasToken, AnchorToken, TagToken)) for token in yaml.scan(text)):
            raise SkillLoadError("yaml_features_not_allowed")
        loader = _StrictSafeLoader(text)
        try:
            value = loader.get_single_data()
        finally:
            loader.dispose()
    except SkillLoadError:
        raise
    except yaml.YAMLError as exc:
        raise SkillLoadError("invalid_yaml") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SkillLoadError("invalid_manifest")
    unknown = set(value) - _MANIFEST_KEYS
    missing = _REQUIRED_KEYS - set(value)
    if unknown:
        raise SkillLoadError("unknown_manifest_key")
    if missing:
        raise SkillLoadError("missing_manifest_key")
    return value


def _string(value: Any, *, code: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise SkillLoadError(code)
    return value


def _string_list(value: Any, *, code: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SkillLoadError(code)
    if len(set(value)) != len(value):
        raise SkillLoadError(f"duplicate_{code}")
    return value


def _resource_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise SkillLoadError("invalid_resource_path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise SkillLoadError("invalid_resource_path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] not in {"references", "templates"}
        or relative.suffix.lower() != ".md"
    ):
        raise SkillLoadError("invalid_resource_path")
    return relative


def _validate_bundle_tree(bundle: Path, declared: set[str]) -> None:
    expected_dirs = {PurePosixPath(item).parts[0] for item in declared}
    try:
        entries = sorted(bundle.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise SkillLoadError("unreadable_bundle") from exc
    for entry in entries:
        if entry.name == "SKILL.md":
            continue
        if entry.is_symlink():
            raise SkillLoadError("symlink_not_allowed")
        if entry.name == "scripts":
            raise SkillLoadError("scripts_not_allowed")
        if entry.name not in expected_dirs or not entry.is_dir():
            raise SkillLoadError("undeclared_bundle_entry")
        try:
            children = sorted(entry.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise SkillLoadError("unreadable_bundle") from exc
        for child in children:
            relative = f"{entry.name}/{child.name}"
            if child.is_symlink():
                raise SkillLoadError("symlink_not_allowed")
            if not child.is_file() or relative not in declared:
                raise SkillLoadError("undeclared_bundle_entry")


def load_skill_bundle(
    bundle: Path,
    *,
    limits: SkillLimits,
    known_tools: frozenset[str],
) -> SkillSpec:
    """Validate and load one bundle without executing or resolving content."""

    try:
        bundle_info = bundle.lstat()
    except OSError as exc:
        raise SkillLoadError("unreadable_bundle") from exc
    if stat.S_ISLNK(bundle_info.st_mode):
        raise SkillLoadError("symlink_not_allowed")
    if not stat.S_ISDIR(bundle_info.st_mode):
        raise SkillLoadError("invalid_bundle_type")

    skill_text = _read_text_file(
        bundle / "SKILL.md",
        max_chars=limits.max_bundle_chars,
        oversize_code="bundle_too_large",
    )
    manifest_text, instructions = _split_skill_file(skill_text)
    if len(instructions) > limits.max_instruction_chars:
        raise SkillLoadError("instructions_too_large")
    manifest = _load_manifest(manifest_text)

    skill_id = _string(manifest["id"], code="invalid_id", max_chars=64)
    if not _SKILL_ID.fullmatch(skill_id) or skill_id != bundle.name:
        raise SkillLoadError("invalid_id")
    name = _string(manifest["name"], code="invalid_name", max_chars=80)
    description = _string(
        manifest["description"],
        code="invalid_description",
        max_chars=240,
    )
    version = manifest["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SkillLoadError("invalid_version")

    allowed_tools = _string_list(manifest["allowed_tools"], code="allowed_tools")
    if set(allowed_tools) - known_tools:
        raise SkillLoadError("unknown_tool")
    modes = _string_list(manifest["supported_modes"], code="supported_modes")
    if not modes or set(modes) - _SUPPORTED_MODES:
        raise SkillLoadError("invalid_supported_modes")

    raw_resources = manifest.get("resources", [])
    resources_list = _string_list(raw_resources, code="resources")
    normalized_paths = [_resource_path(item) for item in resources_list]
    declared = {path.as_posix() for path in normalized_paths}
    _validate_bundle_tree(bundle, declared)

    total_chars = len(skill_text)
    resources: list[SkillResource] = []
    for relative in sorted(normalized_paths, key=lambda item: item.as_posix()):
        content = _read_text_file(
            bundle.joinpath(*relative.parts),
            max_chars=limits.max_resource_chars,
            oversize_code="resource_too_large",
        )
        total_chars += len(content)
        if total_chars > limits.max_bundle_chars:
            raise SkillLoadError("bundle_too_large")
        resources.append(SkillResource(path=relative.as_posix(), content=content))

    digest = hashlib.sha256()

    def add_digest_part(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add_digest_part("SKILL.md")
    add_digest_part(skill_text)
    for resource in resources:
        add_digest_part(resource.path)
        add_digest_part(resource.content)

    return SkillSpec(
        id=skill_id,
        name=name,
        description=description,
        version=version,
        digest=digest.hexdigest(),
        instructions=instructions,
        allowed_tools=frozenset(allowed_tools),
        supported_modes=frozenset(cast(list[SkillMode], modes)),
        resources=tuple(resources),
    )


__all__ = ["SkillLimits", "SkillLoadError", "load_skill_bundle"]
