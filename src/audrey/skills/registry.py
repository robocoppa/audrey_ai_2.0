"""Atomic in-memory registry for validated local skill bundles."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from audrey.metrics import skill_requests_total, skill_selection_seconds
from audrey.skills.loader import SkillLimits, SkillLoadError, load_skill_bundle
from audrey.skills.models import (
    ResolvedSkill,
    SkillCatalogEntry,
    SkillIssue,
    SkillMode,
    SkillRecord,
    SkillRegistrySnapshot,
    SkillSelectionReason,
)
from audrey.skills.selection import SkillEvidence, select_automatic_skill


@dataclass(frozen=True, slots=True)
class _RegistryState:
    records: Mapping[str, SkillRecord]
    issues: tuple[SkillIssue, ...]


_EMPTY_STATE = _RegistryState(records=MappingProxyType({}), issues=())
_SAFE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillSelectionError(ValueError):
    """Safe, structured failure raised before a skill reaches a model."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        skill_id: str = "",
        unavailable_tools: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.skill_id = skill_id
        self.unavailable_tools = unavailable_tools

    def detail(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.code, "message": str(self)}
        if self.skill_id:
            out["skill"] = self.skill_id
        if self.unavailable_tools:
            out["unavailable_tools"] = list(self.unavailable_tools)
        return out


def skill_mode_for_virtual_model(virtual_model: str) -> SkillMode:
    if virtual_model == "audrey_fast":
        return "fast"
    if virtual_model in {
        "audrey_deep",
        "audrey_cloud",
        "audrey_local",
        "audrey_research",
    }:
        return "deep"
    return "auto"


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
        auto_select: bool = False,
    ) -> None:
        self.enabled = enabled
        self.auto_select = auto_select
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
            auto_select=bool(config.get("auto_select", False)),
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

    def resolve(
        self,
        *,
        explicit_skill: str | None,
        virtual_model: str,
        mode: SkillMode,
    ) -> ResolvedSkill | None:
        """Resolve and publish bounded selection telemetry."""

        reason = "request" if explicit_skill else (
            "virtual_model" if virtual_model in self._virtual_models else "none"
        )
        started = time.perf_counter()
        try:
            resolved = self._resolve(
                explicit_skill=explicit_skill,
                virtual_model=virtual_model,
                mode=mode,
            )
        except SkillSelectionError as exc:
            label = exc.skill_id if exc.skill_id in self._state.records else "unknown"
            skill_requests_total.labels(
                skill=label, reason=reason, outcome=exc.code,
            ).inc()
            raise
        else:
            skill_requests_total.labels(
                skill=resolved.spec.id if resolved is not None else "none",
                reason=resolved.reason if resolved is not None else reason,
                outcome="selected" if resolved is not None else "none",
            ).inc()
            return resolved
        finally:
            skill_selection_seconds.labels(reason=reason).observe(
                time.perf_counter() - started
            )

    def _resolve(
        self,
        *,
        explicit_skill: str | None,
        virtual_model: str,
        mode: SkillMode,
    ) -> ResolvedSkill | None:
        """Resolve one request against one immutable registry snapshot."""

        state = self._state
        mapped_id = self._virtual_models.get(virtual_model)
        if explicit_skill and mapped_id and explicit_skill != mapped_id:
            raise SkillSelectionError(
                "skill_conflict",
                f"Skill {explicit_skill!r} conflicts with virtual model "
                f"{virtual_model!r}, which selects {mapped_id!r}.",
                skill_id=explicit_skill,
            )
        skill_id = explicit_skill or mapped_id
        if not skill_id:
            return None
        reason: SkillSelectionReason = (
            "request" if explicit_skill else "virtual_model"
        )
        if not self.enabled:
            if explicit_skill:
                raise SkillSelectionError(
                    "skills_disabled",
                    "Explicit skill selection is disabled on this deployment.",
                    skill_id=skill_id,
                )
            return None
        record = state.records.get(skill_id)
        if record is None:
            if explicit_skill:
                available = sorted(state.records)
                suffix = f" Available skills: {available}." if available else ""
                raise SkillSelectionError(
                    "unknown_skill",
                    f"Unknown skill {skill_id!r}.{suffix}",
                    skill_id=skill_id,
                )
            return None
        if record.unavailable_tools:
            if explicit_skill:
                raise SkillSelectionError(
                    "skill_unavailable",
                    f"Skill {skill_id!r} is unavailable because required "
                    "tools are missing.",
                    skill_id=skill_id,
                    unavailable_tools=tuple(sorted(record.unavailable_tools)),
                )
            return None
        if mode not in record.spec.supported_modes:
            if explicit_skill:
                raise SkillSelectionError(
                    "skill_mode_unsupported",
                    f"Skill {skill_id!r} does not support {mode!r} mode.",
                    skill_id=skill_id,
                )
            return None
        return ResolvedSkill(spec=record.spec, reason=reason)

    def resolve_auto(
        self,
        *,
        prompt: str,
        files: tuple[SkillEvidence, ...],
        virtual_model: str,
        mode: SkillMode,
    ) -> ResolvedSkill | None:
        """Native opt-in selection; explicit and mapped resolution runs first.

        Unavailable or unsupported skills abstain instead of failing ordinary
        chat. Compatibility/API handlers never call this method.
        """

        if not self.enabled or not self.auto_select:
            return None
        if virtual_model in self._virtual_models or virtual_model.startswith(
            "audrey_passthrough/"
        ):
            return None
        started = time.perf_counter()
        resolved = None
        try:
            skill_id = select_automatic_skill(prompt, files)
            record = self._state.records.get(skill_id) if skill_id else None
            if record and not record.unavailable_tools and mode in record.spec.supported_modes:
                resolved = ResolvedSkill(spec=record.spec, reason="automatic")
            return resolved
        finally:
            skill_requests_total.labels(
                skill=resolved.spec.id if resolved else "none",
                reason="automatic",
                outcome="selected" if resolved else "none",
            ).inc()
            skill_selection_seconds.labels(reason="automatic").observe(
                time.perf_counter() - started
            )

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


__all__ = [
    "SkillRegistry",
    "SkillSelectionError",
    "skill_mode_for_virtual_model",
]
