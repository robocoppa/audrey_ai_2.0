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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ValidationError


def _to_str(v: Any) -> Any:
    """Coerce a scalar id to str. Models emit `source_ids` and ids as INTEGERS
    (`[1]`, `2`) about a third of the time — the schema wants strings, and
    without this Pydantic raises ValidationError and the whole worker ledger is
    discarded (the observed 2/3-workers-drop bug). Leave non-scalars alone so a
    genuinely wrong shape still fails."""
    if isinstance(v, (int, float)):
        return str(v)
    return v


def _to_str_list(v: Any) -> Any:
    """Coerce each element of an id list to str (see `_to_str`)."""
    if isinstance(v, list):
        return [str(x) if isinstance(x, (int, float)) else x for x in v]
    return v


# Reusable id types that tolerate int-or-str from the model.
StrId = Annotated[str, BeforeValidator(_to_str)]
StrIdList = Annotated[list[str], BeforeValidator(_to_str_list)]

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

# The set form, for tolerant validation (see _norm_source_type).
_SOURCE_TYPES = frozenset(
    ["official", "primary_paper", "scholarly", "reference",
     "news", "company_claim", "blog", "unknown"]
)

Risk = Literal["low", "medium", "high"]

Verdict = Literal[
    "supported",       # the source(s) back the claim as stated
    "unsupported",     # no source actually supports it (e.g. "surviving: Conics"
                       # when Conics is lost) — writer omits it
    "conflicting",     # sources or other claims contradict it
    "needs_hedge",     # plausible but the exact form isn't earned — soften
    "irrelevant",      # not load-bearing; ignore
]


def _norm_source_type(v: Any) -> Any:
    """Map an unrecognized source_type to 'unknown' instead of failing the whole
    ledger. Models occasionally emit values outside the enum (e.g. 'wikipedia',
    'web'); a single bad type shouldn't discard every source the worker found."""
    if isinstance(v, str) and v not in _SOURCE_TYPES:
        return "unknown"
    return v


class Source(BaseModel):
    # id optional: models routinely omit it (writing title/url first). A
    # required id with no default discarded the whole worker ledger (the
    # 2/3-drop bug). Missing ids are backfilled positionally in the parser.
    id: StrId = ""
    title: str = ""
    url: str = ""  # not HttpUrl: models emit imperfect URLs and a hard validator
                   # would reject an otherwise-usable source. We sanity-check
                   # shape downstream when rendering "Sources used", not here.
    source_type: Annotated[SourceType, BeforeValidator(_norm_source_type)] = "unknown"
    supports: StrIdList = []  # claim ids this source backs


class Claim(BaseModel):
    id: StrId = ""          # optional + backfilled — see Source.id
    text: str = ""
    source_ids: StrIdList = []
    risk: Risk = "medium"
    needs_hedge: bool = False
    hedge_reason: str | None = None


class ResearchResult(BaseModel):
    summary_notes: str = ""
    claims: list[Claim] = []
    sources: list[Source] = []
    unresolved_questions: list[str] = []


class ClaimCheck(BaseModel):
    claim_id: StrId
    verdict: Verdict
    corrected_text: str | None = None
    notes: str = ""


class FactCheckResult(BaseModel):
    checks: list[ClaimCheck] = []
    fatal_errors: list[str] = []


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively replace every {"$ref": "#/$defs/X"} with a copy of defs[X].

    Ollama's `format` field feeds the schema to the model's constrained decoder,
    and `$ref`/`$defs` resolution is inconsistently supported across models —
    the big cloud models (qwen3.5, glm-5.2) returned unusable JSON for our
    nested schemas while deepseek and the local qwen3.6 handled them. Inlining
    the refs into a self-contained schema makes structured output work uniformly.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            target = defs.get(name, {})
            return _inline_refs(dict(target), defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def inlined_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A `$ref`-free JSON schema for `model`, safe to pass to Ollama `format`."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _inline_refs(schema, defs)


def _strip_fence(s: str) -> str:
    """Strip a leading ```/```json fence and any trailing ``` from a reply."""
    s = s.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing fence.
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _backfill_ids(r: ResearchResult) -> ResearchResult:
    """Give every claim/source a stable id when the model omitted one, and drop
    claims with no text. Models often write the content but skip `id`; we assign
    positional ids (`c1`, `c2`, `s1`, …) so downstream id-linkage (source_ids,
    claim_id verdicts) still has something to reference. Pure (returns r mutated;
    r is freshly parsed, not shared)."""
    r.claims = [c for c in r.claims if c.text.strip()]
    for i, c in enumerate(r.claims, 1):
        if not c.id:
            c.id = f"c{i}"
    for i, s in enumerate(r.sources, 1):
        if not s.id:
            s.id = f"s{i}"
    return r


def _extract_json(raw: str) -> str | None:
    """Best-effort: pull a JSON value (object OR array) out of a model reply.

    Models pinned to a JSON schema usually return clean JSON, but observed
    failures show two real cases: the reply is wrapped in a ```json fence, and
    (for list-shaped results) the model returns a bare top-level array instead
    of the wrapping object. We strip the fence, then take the outermost {...} or
    [...] span — whichever starts first. The caller normalizes a bare array.
    """
    s = _strip_fence(raw)
    if not s:
        return None
    obj_start, arr_start = s.find("{"), s.find("[")
    candidates = [(i, c) for i, c in ((obj_start, "}"), (arr_start, "]")) if i != -1]
    if not candidates:
        return None
    start, close = min(candidates)  # whichever bracket opens first
    end = s.rfind(close)
    if end > start:
        return s[start:end + 1]
    return None


def parse_research_result(raw: str) -> ResearchResult | None:
    """Parse a researcher reply into a ResearchResult, or None if unusable.

    Never raises — a None return tells the caller to fall back to prose handling
    for this worker. Tolerates code fences, surrounding prose, and a bare
    top-level array (some models return just the claims list).
    """
    candidate = _extract_json(raw)
    if candidate is None:
        return None
    try:
        # strict=False: models put multi-line prose in string values with raw
        # (unescaped) newlines, which strict json.loads rejects as a control
        # character mid-body — the real cause of "valid-looking but unusable".
        data = json.loads(candidate, strict=False)
        # A bare array → assume it's the claims list and wrap it.
        if isinstance(data, list):
            data = {"claims": data}
        result = ResearchResult.model_validate(data)
        return _backfill_ids(result)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        log.info("ledger.parse_research_result: unusable model output — %s: %s", type(e).__name__, e)
        return None


def parse_factcheck_result(raw: str) -> FactCheckResult | None:
    """Parse a fact-checker reply into a FactCheckResult, or None if unusable.

    Never raises (see `parse_research_result`). Tolerates a bare top-level array
    of checks — observed on the box, the model returns `[{...}, ...]` rather than
    `{"checks": [...]}`.
    """
    candidate = _extract_json(raw)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate, strict=False)  # see parse_research_result
        if isinstance(data, list):
            data = {"checks": data}
        return FactCheckResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        log.info("ledger.parse_factcheck_result: unusable model output — %s: %s", type(e).__name__, e)
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
    "inlined_schema",
    "parse_research_result",
    "parse_factcheck_result",
]
