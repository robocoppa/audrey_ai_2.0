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
the prior prose behaviour for that request. See `docs/campaign-2/phase-26-research-claim-ledger.md`.
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

# Source types that, on their own, earn a plain statement (independent of the
# claimant). `news`/`blog`/`unknown` do not — they fall through to hedging.
# Used by `hedge_policy` (rule 4) and by `_demote_urlless_authority`, which
# strips membership from any source that carries no URL. Defined here rather
# than beside `hedge_policy` because the normalizer needs it first.
_AUTHORITATIVE_SOURCES = frozenset(
    ["official", "primary_paper", "scholarly", "reference"]
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


# Fields the DECODER must emit, keyed by the schema object they belong to.
#
# This is deliberately NOT the same thing as the Pydantic model's required
# fields, and the two must not be merged. Parsing stays maximally tolerant —
# every field defaulted, so one `url: null` or one missing `id` can never
# discard a whole worker's ledger (the 2/3-drop bug). Generation is the
# opposite problem: `model_json_schema()` emits a field WITH a default as
# optional, so the schema we hand Ollama `format` carried no `required` list at
# any level, and a constrained decoder is free to close a `Claim` object right
# after `text`.
#
# 2026-08-13, `current-rust-async`: one worker (deepseek) returned ~41 claims
# and 5 real cited URLs with `source_ids` empty on essentially all of them,
# while its sibling workers linked theirs normally. That all-or-nothing shape is
# the signature of a decode path, not of per-claim judgement — once a model
# settles on the shorter object shape early in a long array it keeps emitting
# it, because the grammar allows it. The structuring prompt has said "LINK EACH
# CLAIM" in capitals for weeks; prompt prose does not outrank the grammar.
#
# What this buys and what it does NOT: requiring the key forces the model to
# emit `"source_ids": [` and make the linkage decision explicitly, per claim,
# with the sources array in context. It cannot force a NON-empty array — an
# honestly unsourced claim can still emit `[]`, which is correct behaviour.
# Judge it by `UNLINKED-LEDGER` in the structuring log (claims and sources both
# present, zero links), which is a presence/absence reading, not a rate.
#
# `id` is deliberately NOT required anywhere: models routinely omit it and
# `_backfill_ids` assigns positional ids, so requiring it only invites a
# fabricated or duplicated one. `supports` is derived by `backfill_supports`.
_REQUIRED_FOR_DECODE: dict[str, tuple[str, ...]] = {
    "ResearchResult": ("claims", "sources"),
    "Claim": ("text", "source_ids", "risk"),
    "Source": ("title", "url", "source_type"),
    "FactCheckResult": ("checks",),
    "ClaimCheck": ("claim_id", "verdict"),
}


def _require_for_decode(obj: Any, name: str) -> Any:
    """Stamp `required` onto one inlined schema object, per `_REQUIRED_FOR_DECODE`.

    Also drops `default` from the properties it requires: a property that says
    both "you must emit me" and "here is my value if you don't" hands the
    decoder two different stories, and the default is the one we're trying to
    stop the model taking."""
    fields = _REQUIRED_FOR_DECODE.get(name)
    if not fields or not isinstance(obj, dict):
        return obj
    props = obj.get("properties")
    if not isinstance(props, dict):
        return obj
    present = [f for f in fields if f in props]
    if not present:
        return obj
    obj["required"] = present
    for f in present:
        props[f] = {k: v for k, v in props[f].items() if k != "default"}
    return obj


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
            # Stamp `required` here, where the $defs name is still known — the
            # inlined copy loses every other trace of which model it came from.
            return _require_for_decode(_inline_refs(dict(target), defs), name)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def inlined_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A `$ref`-free JSON schema for `model`, safe to pass to Ollama `format`.

    Nested objects get their `required` list stamped on during inlining; the
    root is not a `$ref`, so it is stamped here by model name. See
    `_REQUIRED_FOR_DECODE` for why generation is constrained where parsing
    is not."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _require_for_decode(_inline_refs(schema, defs), model.__name__)


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


def backfill_supports(r: ResearchResult) -> ResearchResult:
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


# Unambiguous authoritative domains, matched against a source URL's host when the
# model left `source_type: unknown`. Regex on the host (substring of the netloc)
# rather than urlparse to match this module's `re`-only idiom. Kept deliberately
# narrow — only domains whose *type* is not in doubt, so we never upgrade a blog
# that merely quotes a paper. 2026-07-14 writer-A/B trace: researchers tagged the
# NeurIPS PDF and arxiv.org/abs/1706.03762 for "Attention Is All You Need" as
# `unknown`, so `hedge_policy` rule 5 hedged settled facts ("softmax is applied
# row-wise") for lack of an authoritative source type. Repairing the type here —
# next to the other parse-time normalizers — restores plain statement without
# touching the (intentionally conservative) policy logic.
_DOMAIN_SOURCE_TYPES: tuple[tuple[re.Pattern[str], SourceType], ...] = (
    (re.compile(r"(^|\.)arxiv\.org$"), "primary_paper"),
    (re.compile(r"(^|\.)doi\.org$"), "primary_paper"),
    (re.compile(r"(^|\.)papers\.neurips\.cc$"), "primary_paper"),
    (re.compile(r"(^|\.)proceedings\.neurips\.cc$"), "primary_paper"),
    (re.compile(r"(^|\.)pubmed\.ncbi\.nlm\.nih\.gov$"), "primary_paper"),
    (re.compile(r"(^|\.)pmc\.ncbi\.nlm\.nih\.gov$"), "scholarly"),
    (re.compile(r"\.gov$"), "official"),
    (re.compile(r"\.edu$"), "scholarly"),
    (re.compile(r"(^|\.)wikipedia\.org$"), "reference"),
)

_HOST_RE = re.compile(r"^[a-z]+://([^/?#]+)", re.IGNORECASE)


def _host_of(url: str) -> str:
    """Lowercased host of `url` (netloc without port), or '' if unparseable.
    Tolerant like the rest of the parser — a malformed url yields no upgrade
    rather than an error."""
    m = _HOST_RE.match(url.strip())
    if not m:
        return ""
    return m.group(1).split("@")[-1].split(":")[0].lower()


def _upgrade_source_types(r: ResearchResult) -> ResearchResult:
    """Upgrade a source's `source_type` from `unknown` to its real authoritative
    type when its URL host is on an unambiguous authoritative domain.

    Only touches sources the model left as `unknown` — an explicit type the model
    chose is never overridden (it may know something the domain doesn't, e.g. a
    `.gov` page that is really a news release). Pure; runs at parse time so the
    corrected types reach `_source_types_for_claim`/`hedge_policy` downstream.
    See `_DOMAIN_SOURCE_TYPES` for the rationale and the eval that motivated it."""
    for s in r.sources:
        if s.source_type != "unknown" or not s.url:
            continue
        host = _host_of(s.url)
        if not host:
            continue
        for pattern, upgraded in _DOMAIN_SOURCE_TYPES:
            if pattern.search(host):
                s.source_type = upgraded
                break
    return r


def usable_url(url: str) -> bool:
    """A URL we're willing to act on: http(s) with a host.

    Shared by the renderer (which will not show the user an unusable URL) and by
    `_demote_urlless_authority` (which will not let one confer authority). One
    predicate on purpose: a first version of the demotion tested `url.strip()`
    and let `(reference) Aristoxenus … — null` through, because the model emitted
    the four-character STRING "null" rather than a JSON null. `_to_str_or_empty`
    converts a real `null` to "", so the tolerant-parse path never sees it, and a
    non-empty-but-meaningless string satisfies any emptiness test. Requiring a
    scheme and host rejects that, bare titles, and fragments alike."""
    u = (url or "").strip()
    return u.startswith(("http://", "https://")) and len(u) > len("https://")


def _demote_urlless_authority(r: ResearchResult) -> ResearchResult:
    """Strip authoritative status from a source with no URL — it was never
    retrieved, so it cannot ground anything.

    The mirror of `_upgrade_source_types`: that one PROMOTES `unknown` when the
    URL host is unambiguous; this one DEMOTES an authoritative type when there
    is no URL at all. Both exist because `source_type` is emitted by the model.

    2026-08-13, three protocol runs: researchers routinely emit a named
    authority as if it were a fetched source — `(reference) Herodotus, Histories
    — no url`, `(reference) Plutarch (Ancient Writer) — no url`, `(official)
    Meta Llama 4 Family Announcement — no url`. 39 / 34 / 7 such rows across the
    three runs, of which 38 / 28 / 6 backed claims. Because `reference` and
    `official` are in `_AUTHORITATIVE_SOURCES`, `hedge_policy` rule 4 stated
    those claims PLAINLY — so the least-grounded claims in the ledger got the
    most confident phrasing, while claims whose real URLs the structuring pass
    failed to link fell through to rule 5 and hedged. Hedging was close to
    inverted relative to grounding.

    The 2026-07-15 (+75) investigation saw half of this and stopped: it noted
    url-less sources as a driver of OVER-hedging, but every instance it looked
    at was typed `unknown`, which hedges correctly. A url-less source typed
    `reference` does the opposite, and that direction is the dangerous one.

    Demotes to `unknown` rather than dropping the row: the source is still worth
    rendering and reading, it just cannot carry authority. `company_claim` is
    left alone — it is not in the authoritative set, and its whole purpose is to
    force attribution, which stays right whether or not a URL came with it.

    ⚠️ Known cost, judged acceptable: a KB document legitimately retrieved by
    `kb_search` and typed `official` would also be demoted, because the ledger
    has no retrieval-provenance field and URL presence is the only proxy. No
    such row appears in any of the three runs swept (the url-less rows are all
    named authorities or Audrey's own prior memory notes, and a memory note is
    not retrieved evidence either). If KB-grounded research does start
    over-hedging, the fix is to give KB sources a synthetic `kb://<doc>` URL,
    which is the better shape anyway. Pure; runs after `_upgrade_source_types`
    so a promotable URL has already been promoted."""
    for s in r.sources:
        if s.source_type in _AUTHORITATIVE_SOURCES and not usable_url(s.url):
            log.info(
                "ledger: demoting url-less %s source %r to unknown",
                s.source_type, (s.title or s.id)[:60],
            )
            s.source_type = "unknown"
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
        return backfill_supports(
            _demote_urlless_authority(
                _upgrade_source_types(_repair_source_links(_backfill_ids(result)))
            )
        )
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

def hedge_policy(
    claim: Claim,
    source_types: set[SourceType],
    verdict: Verdict | None = None,
) -> HedgeDisposition:
    """Compute how confidently the writer should state `claim`.

    `verdict` is the fact-checker's finding for this claim, or None when the
    claim was never checked (the checker samples — most claims arrive here with
    no verdict at all).

    Order matters — the first matching rule wins:
    1. A vendor's own assertion (`company_claim`) is attributed, never endorsed,
       even if the same fact also has an independent source (the attribution is
       the honest framing). A `supported` verdict does NOT override this: the
       attribution is about WHO says it, which verification doesn't change.
    2. The fact-checker (or researcher) flagged it `needs_hedge` → soften. Also
       not overridden — an explicit "this needs hedging" annotation is a stronger
       signal than a sampled verdict.
    3. High-risk claims hedge unless a strong source carries them — UNLESS the
       fact-checker already checked this claim and returned `supported`.
       ⚠️ Without that exemption the two stages contradict each other: run
       `113119` handed the writer `CONFIRMED: Tokio's latest release is v1.53.1,
       with v1.53.0 released on July 17, 2026` (verified against the official
       GitHub releases page) AND `HEDGE (unless a strong source backs it)` for
       the same sentence, and the writer resolved the conflict by hedging —
       "appears to be v1.53.1 … around mid-July 2026". The label says "unless a
       strong source backs it"; when a verdict exists, that question has been
       ANSWERED, and rule 3 has no business asking it again.
    4. An authoritative claim is stated plainly. "Authoritative" is
       `_AUTHORITATIVE_SOURCES` (defined at the top of the module), and
       `_demote_urlless_authority` has already stripped membership from any
       source that arrived without a URL — a source nobody fetched cannot make
       a claim confident.
    5. Otherwise hedge — the conservative default, which also covers a surviving
       claim whose sources the model never linked (`source_types` empty). A
       `supported` high-risk claim with nothing authoritative behind it lands
       here, and still hedges: verified-but-unsourced is not plain-statement
       material.
    """
    if "company_claim" in source_types:
        return "attribute_to_company"
    if claim.needs_hedge:
        return "hedge"
    if claim.risk == "high" and verdict != "supported":
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
    "usable_url",
    "inlined_schema",
    "parse_research_result",
    "parse_factcheck_result",
]
