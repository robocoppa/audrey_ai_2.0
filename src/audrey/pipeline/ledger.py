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
the prior prose behaviour for that request. See `docs/campaign-2/phase-26-deploy.md`.
"""
from __future__ import annotations

import json
import logging
import re
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


def _to_str_or_empty(v: Any) -> Any:
    """Coerce a missing/None/non-string text field to "". Models emit `url: null`
    for a source they couldn't link (and occasionally an int) — Pydantic rejects
    None for a `str` field, which discarded the whole worker ledger (observed:
    `ValidationError on fields ['sources.0.url', 'sources.4.url']`, dropping a
    worker to 2/3). A blank url is harmless: we sanity-check url shape when
    rendering "Sources used", not here. Leave real strings untouched."""
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return str(v)
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

_VERDICTS = frozenset(
    ["supported", "unsupported", "conflicting", "needs_hedge", "irrelevant"]
)


def _norm_source_type(v: Any) -> Any:
    """Map an unrecognized source_type to 'unknown' instead of failing the whole
    ledger. Models occasionally emit values outside the enum (e.g. 'wikipedia',
    'web'); a single bad type shouldn't discard every source the worker found."""
    if isinstance(v, str) and v.lower() not in _SOURCE_TYPES:
        return "unknown"
    return v.lower() if isinstance(v, str) else v


def _norm_risk(v: Any) -> Any:
    """Map any risk value outside low/medium/high to 'medium'. Models emit
    'High', ints, or descriptive strings ('high - speculative'); one off-enum
    risk per claim was discarding whole ledgers (errors scaled with claim count
    — the 97-validation-error case). Match leniently, default to medium."""
    if isinstance(v, str):
        lv = v.strip().lower()
        for level in ("low", "medium", "high"):
            if lv.startswith(level):
                return level
    return "medium"


class Source(BaseModel):
    # id optional: models routinely omit it (writing title/url first). A
    # required id with no default discarded the whole worker ledger (the
    # 2/3-drop bug). Missing ids are backfilled positionally in the parser.
    id: StrId = ""
    title: Annotated[str, BeforeValidator(_to_str_or_empty)] = ""  # tolerate null
    # not HttpUrl: models emit imperfect URLs and a hard validator would reject
    # an otherwise-usable source. _to_str_or_empty also tolerates `url: null`
    # (model couldn't link the source) — without it, one null url discarded the
    # whole worker ledger. We sanity-check shape when rendering "Sources used".
    url: Annotated[str, BeforeValidator(_to_str_or_empty)] = ""
    source_type: Annotated[SourceType, BeforeValidator(_norm_source_type)] = "unknown"
    supports: StrIdList = []  # claim ids this source backs


class Claim(BaseModel):
    id: StrId = ""          # optional + backfilled — see Source.id
    text: str = ""
    source_ids: StrIdList = []
    risk: Annotated[Risk, BeforeValidator(_norm_risk)] = "medium"
    needs_hedge: bool = False
    hedge_reason: str | None = None


class ResearchResult(BaseModel):
    summary_notes: str = ""
    claims: list[Claim] = []
    sources: list[Source] = []
    unresolved_questions: list[str] = []


def _norm_verdict(v: Any) -> Any:
    """Normalize a verdict to the enum; unknown → 'irrelevant' (ignored
    downstream) rather than discarding the whole fact-check result."""
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in _VERDICTS:
            return lv
    return "irrelevant"


def _norm_fatal_errors(v: Any) -> Any:
    """Coerce each `fatal_errors` entry to a one-line string. Same fail-soft
    intent as `_norm_verdict`: one malformed entry must not sink the whole
    fact-check. 2026-07-08 eval (`bio-euclid`, `hist-library-alexandria`): the
    model returned a correction/conflict OBJECT here — `{'claim_ids': [...],
    ...}` and `{'claim_id': ..., 'conflicting_claim_id': ...}` — against a
    `list[str]` field, so Pydantic rejected the entire FactCheckResult and the
    writer silently got NO CORRECTIONS while a fact-check with real drops/hedges
    was discarded. We flatten dicts to a readable line (preferring a
    message-like field, else a compact `key=value` join) rather than drop them."""
    if not isinstance(v, list):
        return v

    def _flatten(e: Any) -> str:
        if isinstance(e, str):
            return e
        if isinstance(e, dict):
            for key in ("message", "text", "error", "detail", "reason"):
                val = e.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return "; ".join(f"{k}={v}" for k, v in e.items())
        return str(e)

    return [_flatten(e) for e in v]


class ClaimCheck(BaseModel):
    claim_id: StrId = ""
    verdict: Annotated[Verdict, BeforeValidator(_norm_verdict)] = "irrelevant"
    corrected_text: str | None = None
    notes: str = ""


class FactCheckResult(BaseModel):
    checks: list[ClaimCheck] = []
    fatal_errors: Annotated[list[str], BeforeValidator(_norm_fatal_errors)] = []


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
    claims with no text and sources with no content. Models often write the
    content but skip `id`; we assign positional ids (`c1`, `c2`, `s1`, …) so
    downstream id-linkage (source_ids, claim_id verdicts) still has something to
    reference. Pure (returns r mutated; r is freshly parsed, not shared).

    Dropping content-free sources: the fail-soft `Source` schema defaults every
    field (so one `url: null` can't discard the whole worker ledger — the
    2/3-drop guard), but that also lets a source with NO title AND NO url
    validate cleanly. 2026-07-09 trace run (`bio-euclid`, `hist-library-alexandria`):
    the qwen structuring pass emitted rows that rendered as `w2_, untitled — no
    url` (a stray token became the id) — content-free artifacts the too-tolerant
    schema resurrected, inflating the ledger with sources that back nothing and
    read as broken. A source with neither a title nor a url is not a source; drop
    it. A real source with a url but a blank title (the null-title case the
    schema tolerates) still survives — only the entirely-empty rows go."""
    r.claims = [c for c in r.claims if c.text.strip()]
    r.sources = [s for s in r.sources if s.title.strip() or s.url.strip()]
    for i, c in enumerate(r.claims, 1):
        if not c.id:
            c.id = f"c{i}"
    for i, s in enumerate(r.sources, 1):
        if not s.id:
            s.id = f"s{i}"
    return r


def _backfill_supports(r: ResearchResult) -> ResearchResult:
    """Populate each source's `supports` from the claims that cite it, so the
    source→claim index is complete regardless of what the model emitted.

    Models reliably fill `claim.source_ids` (the claim→source direction) but
    routinely leave `source.supports` empty — 2026-07-09 trace run showed
    `supports: none` on EVERY source across every ledger, even where claims
    clearly cited them. Any consumer reading the source→claim direction (or a
    human reading the rendered ledger) then sees no linkage. We invert
    `source_ids` into `supports`: for each claim, add its id to every source it
    cites. Runs AFTER `_repair_source_links`, so `source_ids` already point at
    real source ids; a `source_ids` entry that still matches nothing is skipped
    (no phantom support). Union with any `supports` the model did emit — we only
    add, never drop. Pure (mutates the freshly-parsed r)."""
    by_id = {s.id: s for s in r.sources}
    for c in r.claims:
        for sid in c.source_ids:
            s = by_id.get(sid)
            if s is not None and c.id not in s.supports:
                s.supports.append(c.id)
    return r


def _repair_source_links(r: ResearchResult) -> ResearchResult:
    """Re-point `claim.source_ids` entries that name a source by TITLE, URL,
    a case-variant of its id, or a `src{N}`/`source{N}` variant of an `s{N}`
    id — observed failure shapes. 2026-07-06 eval (`current-rust-async`): the
    model wrote `source_ids: ["Glommio repository (Datadog)"]` while the
    source itself was `{id: "s3", title: "Glommio repository (Datadog)"}`.
    2026-07-07 eval (`bio-euclid`, `bio-pythagoras`): claims cited `S1`
    against a source whose id was `s1` — the id itself is an alias so the
    lowercased lookup repairs case variants. 2026-07-07 second run
    (`bio-euclid`, `tech-transformer-attention`): claims cited `SRC-1` /
    `src_1` against a source whose id was `s1` (the `_backfill_ids` shape) —
    the number-suffix aliases below repair that spelling. Unrepaired, the bad
    refs poison every downstream consumer at once: `_surviving_source_ids`
    keeps garbage (defeating the render-all fallback), and
    `_source_types_for_claim` finds no backing so `hedge_policy` hedges claims
    that actually had authoritative sources. Best-effort and pure: entries
    that match nothing are kept as-is (downstream treats unknown ids as
    no-linkage). Runs after `_backfill_ids` so every source has an id."""
    ids = {s.id for s in r.sources}
    # Exact aliases (id / title / URL), matched on `.strip().lower()`.
    by_alias: dict[str, str] = {}
    # Separate map for the `src{N}`/`source{N}`↔`s{N}` shape, keyed on the
    # punctuation-stripped form so `SRC-1`, `src_1`, `source1` all collapse to
    # `src1`/`source1`. Kept apart from `by_alias` so the aggressive
    # punctuation strip can't collide with a title that happens to reduce to
    # the same letters.
    by_num_ref: dict[str, str] = {}
    for s in r.sources:
        for alias in (s.id, s.title, s.url):
            a = alias.strip().lower()
            if a:
                by_alias.setdefault(a, s.id)
        m = re.fullmatch(r"s(\d+)", s.id.strip().lower())
        if m:
            by_num_ref.setdefault(f"src{m.group(1)}", s.id)
            by_num_ref.setdefault(f"source{m.group(1)}", s.id)

    def _resolve(sid: str) -> str:
        if sid in ids:
            return sid
        hit = by_alias.get(sid.strip().lower())
        if hit:
            return hit
        num = by_num_ref.get(re.sub(r"[\W_]+", "", sid).lower())
        return num or sid

    for c in r.claims:
        c.source_ids = [_resolve(sid) for sid in c.source_ids]
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
        return _backfill_supports(_repair_source_links(_backfill_ids(result)))
    except ValidationError as e:
        # Log the distinct failing fields (not the full multi-error dump) so a
        # recurring strict-field problem is visible at a glance.
        fields = sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()})
        log.info("ledger.parse_research_result: ValidationError on fields %s", fields[:8])
        return None
    except (json.JSONDecodeError, TypeError) as e:
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


# Stage 4: deterministic selective hedging. A pure function (no model, no I/O)
# that maps a claim + the source types backing it to a hedging *disposition* the
# writer applies verbatim — so "DeepSeek released R1 on 2025-01-20" (official,
# low risk → stated plainly) and "Meta claims Maverick beats GPT-4o"
# (company_claim → attributed to the vendor) are deterministic instead of
# left to the writer's confidence.
HedgeDisposition = Literal[
    "state_plainly",          # authoritative + not high-risk: assert directly
    "attribute_to_company",   # a vendor's own claim: name the source, don't endorse
    "hedge",                  # soften — needs_hedge, or no authoritative grounding
    "hedge_or_cite_strongly", # high-risk: soften unless a strong source backs it
]

# Source types that, on their own, earn a plain statement (independent of the
# claimant). `news`/`blog`/`unknown` do not — they fall through to hedging.
_AUTHORITATIVE_SOURCES = frozenset(
    ["official", "primary_paper", "scholarly", "reference"]
)


def hedge_policy(claim: Claim, source_types: set[SourceType]) -> HedgeDisposition:
    """Compute how confidently the writer should state `claim`.

    Order matters — the first matching rule wins:
    1. A vendor's own assertion (`company_claim`) is attributed, never endorsed,
       even if the same fact also has an independent source (the attribution is
       the honest framing).
    2. The fact-checker (or researcher) flagged it `needs_hedge` → soften.
    3. High-risk claims hedge unless a strong source carries them.
    4. An authoritative, non-high-risk claim is stated plainly.
    5. Otherwise hedge — the conservative default, which also covers a surviving
       claim whose sources the model never linked (`source_types` empty).
    """
    if "company_claim" in source_types:
        return "attribute_to_company"
    if claim.needs_hedge:
        return "hedge"
    if claim.risk == "high":
        return "hedge_or_cite_strongly"
    if source_types & _AUTHORITATIVE_SOURCES:
        return "state_plainly"
    return "hedge"


__all__ = [
    "Source",
    "Claim",
    "ResearchResult",
    "ClaimCheck",
    "FactCheckResult",
    "SourceType",
    "Risk",
    "Verdict",
    "HedgeDisposition",
    "hedge_policy",
    "inlined_schema",
    "parse_research_result",
    "parse_factcheck_result",
]
