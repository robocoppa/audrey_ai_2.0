"""Immutable value objects for validated Audrey skill bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SkillMode = Literal["auto", "fast", "deep"]
SkillAvailability = Literal["available", "degraded"]
SkillRegistryStatus = Literal["disabled", "ready", "degraded", "unavailable"]


@dataclass(frozen=True, slots=True)
class SkillResource:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    name: str
    description: str
    version: int
    digest: str
    instructions: str
    allowed_tools: frozenset[str]
    supported_modes: frozenset[SkillMode]
    resources: tuple[SkillResource, ...]


@dataclass(frozen=True, slots=True)
class SkillIssue:
    skill_id: str
    code: str


@dataclass(frozen=True, slots=True)
class SkillRecord:
    spec: SkillSpec
    unavailable_tools: frozenset[str] = frozenset()

    @property
    def availability(self) -> SkillAvailability:
        return "degraded" if self.unavailable_tools else "available"


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    id: str
    name: str
    description: str
    version: int
    supported_modes: tuple[SkillMode, ...]
    availability: SkillAvailability


@dataclass(frozen=True, slots=True)
class SkillRegistrySnapshot:
    enabled: bool
    status: SkillRegistryStatus
    loaded_count: int
    available_count: int
    degraded_count: int
    invalid_count: int
    invalid: tuple[SkillIssue, ...] = ()


__all__ = [
    "SkillAvailability",
    "SkillCatalogEntry",
    "SkillIssue",
    "SkillMode",
    "SkillRecord",
    "SkillRegistrySnapshot",
    "SkillRegistryStatus",
    "SkillResource",
    "SkillSpec",
]
