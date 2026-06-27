"""Research claim/source ledger — the internal contract for `audrey_research`.

Phase 26. The staged research pipeline passes a *structured* ledger between
stages instead of free prose: researchers emit `ResearchResult` (claims +
sources), the fact-checker emits `FactCheckResult` (per-claim verdicts), and the
writer is bound to the supported claims. The ledger is INTERNAL scaffolding the
models reason over — it is not user-facing. The user sees clean prose plus a
short "Sources used" list; the structure exists so the system can prove to
itself where each load-bearing claim came from before the writer makes it read
well.

Everything here is fail-soft: `parse_research_result` / `parse_factcheck_result`
return `None` on malformed model output (never raise), so a stage can degrade to
the prior prose behaviour for that request. See `phase-26-research-ledger-plan.md`.
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

log = logging.getLogger("audrey.ledger")

SourceType = Literal[
    "official",        # vendor docs, the project's own repo/release notes
    "primary_paper",   # the peer-reviewed paper that introduced the thing
    "scholarly",       # university pages, scholarly secondary literature
    "reference",       # Britannica / MacTutor / Stanford Encyclopedia, etc.
    "news",            # journalism
    "company_claim",   # a vendor's own benchmark/marketing assertion — NOT
                       # independent fact; the hedge policy attributes it
    "blog",
    "unknown",
]

Risk = Literal["low", "medium", "high"]

Verdict = Literal[
    "supported",       # the source(s) back the claim as stated
    "unsupported",     # no source actually supports it (e.g. "surviving: Conics"
                       # when Conics is lost) — writer omits it
    "conflicting",     # sources or other claims contradict it
    "needs_hedge",     # plausible but the exact form isn't earned — soften
    "irrelevant",      # not load-bearing; ignore
]


class Source(BaseModel):
    id: str
    title: str
    url: str  # not HttpUrl: models emit imperfect URLs and a hard validator
              # would reject an otherwise-usable source. We sanity-check shape
              # downstream when rendering "Sources used", not here.
    source_type: SourceType
    supports: list[str] = []  # claim ids this source backs


class Claim(BaseModel):
    id: str
    text: str
    source_ids: list[str] = []
    risk: Risk = "medium"
    needs_hedge: bool = False
    hedge_reason: str | None = None


class ResearchResult(BaseModel):
    summary_notes: str = ""
    claims: list[Claim] = []
    sources: list[Source] = []
    unresolved_questions: list[str] = []


class ClaimCheck(BaseModel):
    claim_id: str
    verdict: Verdict
    corrected_text: str | None = None
    notes: str = ""


class FactCheckResult(BaseModel):
    checks: list[ClaimCheck] = []
    fatal_errors: list[str] = []


def _extract_json_object(raw: str) -> str | None:
    """Best-effort: pull the JSON object out of a model reply.

    Models pinned to a JSON schema usually return clean JSON, but not always —
    they may wrap it in ```json fences or add a sentence. We try the whole
    string first, then the outermost {...} span. Returns the candidate string or
    None.
    """
    s = raw.strip()
    if not s:
        return None
    # Strip a leading/trailing code fence if present.
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # Fall back to the outermost brace span.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return None


def parse_research_result(raw: str) -> ResearchResult | None:
    """Parse a researcher reply into a ResearchResult, or None if unusable.

    Never raises — a None return tells the caller to fall back to prose handling
    for this worker. Tolerates code fences and surrounding prose.
    """
    candidate = _extract_json_object(raw)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
        return ResearchResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        log.info("ledger.parse_research_result: unusable model output (%s)", type(e).__name__)
        return None


def parse_factcheck_result(raw: str) -> FactCheckResult | None:
    """Parse a fact-checker reply into a FactCheckResult, or None if unusable.

    Never raises (see `parse_research_result`).
    """
    candidate = _extract_json_object(raw)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
        return FactCheckResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        log.info("ledger.parse_factcheck_result: unusable model output (%s)", type(e).__name__)
        return None


__all__ = [
    "Source",
    "Claim",
    "ResearchResult",
    "ClaimCheck",
    "FactCheckResult",
    "SourceType",
    "Risk",
    "Verdict",
    "parse_research_result",
    "parse_factcheck_result",
]
