"""Atomic in-memory registry for validated local skill bundles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from audrey.skills.loader import SkillLimits, SkillLoadError, load_skill_bundle
from audrey.skills.models import (
    SkillCatalogEntry,
    SkillIssue,
    SkillRecord,
    SkillRegistrySnapshot,
)


@dataclass(frozen=True, slots=True)
class _RegistryState:
    records: Mapping[str, SkillRecord]
    issues: tuple[SkillIssue, ...]


_EMPTY_STATE = _RegistryState(records=MappingProxyType({}), issues=())
_SAFE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillRegistry:
    """Own validated specs and replace the whole view on rediscovery."""

    def __init__(
        self,
        *,
        enabled: bool,
        roots: tuple[Path, ...],
        limits: SkillLimits,
        known_tools: frozenset[str],
        available_tools: frozenset[str],
        virtual_models: Mapping[str, str] | None = None,
    ) -> None:
        self.enabled = enabled
        self._roots = roots
        self._limits = limits
        self._known_tools = known_tools
        self._available_tools = available_tools
        self._virtual_models = MappingProxyType(dict(virtual_models or {}))
        self._state = _EMPTY_STATE
        if enabled:
            self.rediscover()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        known_tools: frozenset[str],
        available_tools: frozenset[str],
    ) -> SkillRegistry:
        return cls(
            enabled=bool(config.get("enabled", False)),
            roots=tuple(Path(value) for value in config.get("roots", ["/app/skills"])),
            limits=SkillLimits(
                max_instruction_chars=int(
                    config.get("max_instruction_chars", 12_000)
                ),
                max_resource_chars=int(config.get("max_resource_chars", 12_000)),
                max_bundle_chars=int(config.get("max_bundle_chars", 32_000)),
            ),
            known_tools=known_tools,
            available_tools=available_tools,
            virtual_models=config.get("virtual_models", {}),
        )

    def rediscover(
        self,
        *,
        available_tools: frozenset[str] | None = None,
    ) -> SkillRegistrySnapshot:
        """Build a complete replacement before one atomic reference swap."""

        if available_tools is not None:
            self._available_tools = available_tools
        current_tools = self._available_tools
        if not self.enabled:
            return self.snapshot()

        records: dict[str, SkillRecord] = {}
        issues: list[SkillIssue] = []
        duplicates: set[str] = set()
        for root_index, root in enumerate(self._roots):
            try:
                root.lstat()
                if root.is_symlink() or not root.is_dir():
                    raise OSError
                bundles = sorted(root.iterdir(), key=lambda path: path.name)
            except OSError:
                issues.append(
                    SkillIssue(skill_id=f"root-{root_index}", code="root_unavailable")
                )
                continue
            for bundle_index, bundle in enumerate(bundles):
                if bundle.name == ".gitignore":
                    continue
                candidate_id = (
                    bundle.name
                    if _SAFE_ID.fullmatch(bundle.name)
                    else f"bundle-{root_index}-{bundle_index}"
                )
                try:
                    spec = load_skill_bundle(
                        bundle,
                        limits=self._limits,
                        known_tools=self._known_tools,
                    )
                except SkillLoadError as exc:
                    issues.append(SkillIssue(skill_id=candidate_id, code=exc.code))
                    continue
                if spec.id in duplicates:
                    continue
                if spec.id in records:
                    records.pop(spec.id)
                    duplicates.add(spec.id)
                    issues.append(
                        SkillIssue(skill_id=spec.id, code="duplicate_skill_id")
                    )
                    continue
                records[spec.id] = SkillRecord(
                    spec=spec,
                    unavailable_tools=spec.allowed_tools - current_tools,
                )

        issue_ids = {issue.skill_id for issue in issues}
        for skill_id in sorted(set(self._virtual_models.values()) - set(records)):
            if skill_id not in issue_ids:
                issues.append(
                    SkillIssue(
                        skill_id=skill_id,
                        code="mapped_skill_unavailable",
                    )
                )

        self._state = _RegistryState(
            records=MappingProxyType(records),
            issues=tuple(sorted(issues, key=lambda item: (item.skill_id, item.code))),
        )
        return self.snapshot()

    def get(self, skill_id: str) -> SkillRecord | None:
        return self._state.records.get(skill_id)

    def resolve_virtual_model(self, virtual_model: str) -> SkillRecord | None:
        """Return one usable mapped bundle; unavailable mappings fail closed."""

        if not self.enabled:
            return None
        skill_id = self._virtual_models.get(virtual_model)
        record = self._state.records.get(skill_id) if skill_id else None
        if record is None or record.unavailable_tools:
            return None
        return record

    def refresh_availability(
        self,
        available_tools: frozenset[str],
    ) -> SkillRegistrySnapshot:
        """Re-evaluate tool availability without rereading bundle files."""

        self._available_tools = available_tools
        if not self.enabled:
            return self.snapshot()
        state = self._state
        self._state = _RegistryState(
            records=MappingProxyType({
                skill_id: SkillRecord(
                    spec=record.spec,
                    unavailable_tools=record.spec.allowed_tools - available_tools,
                )
                for skill_id, record in state.records.items()
            }),
            issues=state.issues,
        )
        return self.snapshot()

    def catalog(self) -> tuple[SkillCatalogEntry, ...]:
        state = self._state
        return tuple(
            SkillCatalogEntry(
                id=record.spec.id,
                name=record.spec.name,
                description=record.spec.description,
                version=record.spec.version,
                supported_modes=tuple(sorted(record.spec.supported_modes)),
                availability=record.availability,
            )
            for record in sorted(
                state.records.values(),
                key=lambda item: item.spec.id,
            )
        )

    def snapshot(self) -> SkillRegistrySnapshot:
        state = self._state
        records = tuple(state.records.values())
        available_count = sum(not record.unavailable_tools for record in records)
        degraded_count = len(records) - available_count
        invalid_count = len(state.issues)
        if not self.enabled:
            status = "disabled"
        elif invalid_count and not records:
            status = "unavailable"
        elif invalid_count or degraded_count:
            status = "degraded"
        else:
            status = "ready"
        return SkillRegistrySnapshot(
            enabled=self.enabled,
            status=status,
            loaded_count=len(records),
            available_count=available_count,
            degraded_count=degraded_count,
            invalid_count=invalid_count,
            invalid=state.issues,
        )


__all__ = ["SkillRegistry"]
