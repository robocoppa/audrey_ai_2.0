"""Conservative English file intent rules; no model calls or content inspection.

Adapted from the retained English selection study. Automatic selection requires
an affirmative request about resolvable evidence of one supported kind. This is
not a general intent classifier: unclear targets simply retain ordinary chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkillEvidence:
    """Server-verified metadata, never a filename supplied as authorization."""

    filename: str
    kind: str
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class _Request:
    prompt: str
    files: tuple[SkillEvidence, ...]


_OWN_REFERENCE = re.compile(
    r"\b(?:attach(?:ed|ment|ments)?|upload(?:ed|s)?|my (?:file|document|pdf|video|recording)s?"
    r"|(?:this|these|that|those|the|project) (?:file|document|pdf|video|recording)s?)\b",
    re.I,
)
_DOCUMENT = re.compile(
    r"\b(?:documents?|pdfs?|reports?|contracts?|invoices?|spreadsheets?)\b", re.I
)
_VIDEO = re.compile(r"\b(?:videos?|recordings?|clips?|footage)\b", re.I)
_INTENT = re.compile(
    r"\b(?:summari[sz]e|summary|analy[sz]e|compare|describe|explain|extract|review|outline|find|count|read)\b"
    r"|\bwhat\b.{0,80}\b(?:say|says|show|shows|contain|contains|main|takeaway|happens|happened)\b"
    r"|\b(?:what|which|why|who|when|where|how much|how many)\b"
    r"|\bdoes\b.{0,80}\b(?:occur|appear)\b",
    re.I,
)
# Match English requests conservatively; unclear forms retain ordinary chat.
_QUOTED = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))*'|\"(?:\\.|[^\"\\])*\"|“[^”\n]*”|‘(?:[^’\n]|(?<=\w)’(?=\w))*’"
)
_FILE_TOKEN = re.compile(r"\b[\w.-]+\.[a-z][a-z0-9]{0,15}(?!\w)", re.I)
_ACTION = (
    r"(?:summari[sz]e|analy[sz]e|compare|describe|explain|extract|review|outline|find|count|read)"
)
_CLAUSE_BREAK = re.compile(
    rf"[;\n]|[.!?](?=\s|$)|\b(?:and|but|then)\s+(?=(?:not\b|do not\b|don['’]t\b|never\b|ignore\b|{_ACTION}\b|(?:in|from)\b|at\s+\d{{1,2}}:\d{{2}}))"
    rf"|,\s*(?=(?:not\b|do not\b|don['’]t\b|never\b|without\b|ignore\b|{_ACTION}\b))",
    re.I,
)
_NEGATED_ACTION = re.compile(
    rf"\b(?:(?:do not|don't)(?: want (?:you )?to)?|never|without|not asking (?:you )?to)\s+"
    rf"(?:read(?:ing)?|open(?:ing)?|access(?:ing)?|watch(?:ing)?|us(?:e|ing)|inspect(?:ing)?|{_ACTION})\b",
    re.I,
)
_EXCLUSION = re.compile(
    r"\b(?:ignore|disregard|unrelated|irrelevant|excluded)\b"
    r"|\b(?:(?:do not|don't)(?: want (?:you )?to)?|never|without)\s+(?:us(?:e|ing)|read(?:ing)?|open(?:ing)?|access(?:ing)?|watch(?:ing)?|analy[sz]e)\b"
    r"|^\s*not\b",
    re.I,
)
_UNRESOLVED = re.compile(
    r"\b(?:cannot|can't|don't|do not) (?:remember|know) which\b"
    r"|\bnot (?:yet )?chosen which\b"
    r"|\bnot sure which\b.{0,60}\b(?:file|document|report|video|upload|recording)s?\b"
    r"|\b(?:unsure|uncertain) which\b.{0,60}\b(?:file|document|report|video|upload|recording)s?\b",
    re.I,
)
_MANAGEMENT = re.compile(
    r"\b(?:rename|delete|move|remove)\b|\b(?:button|sidebar|interface|files page|my files)\b"
    r"|\b(?:group|organize)\b.{0,60}\b(?:conversation|project)s?\b",
    re.I,
)
_METADATA_ONLY = re.compile(r"\b(?:filename|file name|file type|received it)\b", re.I)
_TEXT_TRANSFORM = re.compile(
    r"\b(?:translate|rewrite|copy)\b|\b(?:meaning|phrase|quoted (?:sentence|line|text))\b", re.I
)
_UNSELECTED_TARGET = re.compile(
    r"\bnot (?:yet )?(?:chosen|decided) which\s+(?:file|document|report|video|upload|recording|one)\b",
    re.I,
)
_PROGRAM_TASK = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:write|create|build|implement|debug)\s+"
    r"(?:a |an |the )?(?:(?:python|javascript|shell|bash)\s+)?(?:script|program|function|code|parser)\b",
    re.I,
)
_HYPOTHETICAL_TASK = re.compile(
    r"^\s*(?:imagine|pretend|suppose)\b|\b(?:hypothetical request|example prompt)\b", re.I
)
_CONTENT_ACTION = re.compile(r"\b(?:summari[sz]e|analy[sz]e|compare|extract|review)\b", re.I)
_CONTENT_LOCATION = re.compile(
    r"\b(?:in|from|inside|within|according to)\s+(?:(?:my|the|uploaded|attached)\s+)*$", re.I
)
_CONTENT_PREDICATE = re.compile(
    r"^\s*(?:says?|contains?|states?|describes?|shows?|documents?|explains?|mentions?)\b", re.I
)
_CONTENT_QUESTION = re.compile(r"\b(?:does?|did|why|which)\b|(?:^|,\s*)(?:is|are|was|were)\b", re.I)
_NO_SPECIALIZATION = re.compile(
    r"\b(?:do not|don't|never)\s+(?:activate|select|use)\s+(?:a |any |the )?"
    r"(?:skills?\b|speciali[sz]\w*|grounded-document-analysis\b|video-analysis\b)"
    r"|\b(?:no|without)\s+(?:automatic\s+)?skills?\b",
    re.I,
)


def _request_text(case: _Request) -> tuple[str, bool]:
    """Hide quoted prose/code, while preserving quoted exact file references."""
    quoted_prose = False

    def replace(match: re.Match) -> str:
        nonlocal quoted_prose
        body = match.group().strip("`'\"“”‘’").strip()
        if any(body.casefold() == file.filename.casefold() for file in case.files):
            return match.group()
        if _FILE_TOKEN.fullmatch(body):
            return match.group()  # A quoted missing name still must fail resolution.
        quoted_prose = True
        return " " * len(match.group())

    prompt = re.sub(r"(?m)^\s*>[^\n]*", lambda match: " " * len(match.group()), case.prompt)
    # An unfinished pasted code block is still data, not an active instruction.
    if prompt.count("```") % 2:
        start = prompt.rfind("```")
        prompt = prompt[:start] + " " * (len(prompt) - start)
    return _QUOTED.sub(replace, prompt), quoted_prose


def _file_matches(
    text: str, files: tuple[SkillEvidence, ...]
) -> list[tuple[SkillEvidence, tuple[int, int]]]:
    return [
        (file, match.span())
        for file in files
        for match in re.finditer(
            rf"(?<![\w.-]){re.escape(file.filename)}(?![\w-]|\.\w)", text, re.I
        )
    ]


def _mask_filenames(text: str, files: tuple[SkillEvidence, ...]) -> str:
    spans = [span for _, span in _file_matches(text, files)]
    spans.extend(match.span() for match in _FILE_TOKEN.finditer(text))
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars).replace("’", "'")


def _request_clauses(text: str, files: tuple[SkillEvidence, ...]) -> list[str]:
    """Keep a file-location prefix with its action across an ordinary comma."""
    clauses = []
    start = 0
    file_spans = [span for _, span in _file_matches(text, files)]
    for split in _CLAUSE_BREAK.finditer(text):
        if any(left <= split.start() < right for left, right in file_spans):
            continue  # Punctuation in an exact filename is not a clause break.
        if split.group().startswith(",") and re.match(
            rf"\s*{_ACTION}\b", text[split.end() :], re.I
        ):
            prefix_text = text[start : split.start()]
            prefix = _mask_filenames(prefix_text, files)
            source_prefix = any(
                _CONTENT_LOCATION.search(prefix_text[: span[0]])
                for _, span in _file_matches(prefix_text, files)
            )
            if source_prefix and not (_EXCLUSION.search(prefix) or _NEGATED_ACTION.search(prefix)):
                continue
        clauses.append(text[start : split.start()])
        start = split.end()
    clauses.append(text[start:])
    return clauses


def _file_content_request(clause: str, words: str, matches: list) -> bool:
    """Distinguish document content about an operation from performing it."""
    # A location question can mention a quoted instruction without asking us
    # to rewrite that instruction. The quote itself remains masked data.
    intent = bool(_INTENT.search(words))
    for _, (start, end) in matches:
        if (intent or _CONTENT_QUESTION.search(words)) and (
            _CONTENT_LOCATION.search(clause[:start]) or _CONTENT_PREDICATE.search(clause[end:])
        ):
            return True
    return bool(matches and intent and _CONTENT_ACTION.search(words))


def _select(case: _Request) -> str | None:
    """Resolve affirmative target clauses; excluded files never replace targets."""
    text, quoted_prose = _request_text(case)
    lexical = _mask_filenames(text, case.files)
    known_targets = _file_matches(text, case.files)
    alternative = bool(re.search(r"\b(?:one of|either)\b.{0,160}\bor\b", text, re.I))
    if _UNSELECTED_TARGET.search(lexical) or (
        _UNRESOLVED.search(lexical) and (not known_targets or alternative)
    ):
        return None
    if _NO_SPECIALIZATION.search(lexical):
        return None

    references: list[SkillEvidence] = []
    recent_refs: list[SkillEvidence] = []
    affirmative_refs: list[SkillEvidence] = []
    excluded: set[SkillEvidence] = set()
    affirmative = []
    negative_analysis = False
    has_management = False
    has_metadata_task = False
    has_transform = False
    has_meta_task = False
    missing_target = False
    affirmative_evidence = False
    for clause in _request_clauses(text, case.files):
        if not clause.strip():
            continue
        matches = _file_matches(clause, case.files)
        clause_files = [file for file, _ in matches]
        words = _mask_filenames(clause, case.files)
        exclusion = bool(_EXCLUSION.search(words))
        negated = bool(_NEGATED_ACTION.search(words))
        content_request = _file_content_request(clause, words, matches)
        if content_request and not negated:
            exclusion = False
        management = bool(_MANAGEMENT.search(words)) and not content_request
        metadata = bool(_METADATA_ONLY.search(words)) and not content_request
        transform = quoted_prose and bool(_TEXT_TRANSFORM.search(words)) and not content_request
        meta_task = bool(_PROGRAM_TASK.search(words) or _HYPOTHETICAL_TASK.search(words))
        has_management |= management
        has_metadata_task |= metadata
        has_transform |= transform
        has_meta_task |= meta_task
        negative_analysis |= negated
        references.extend(clause_files)
        if clause_files:
            recent_refs = clause_files
        if exclusion:
            excluded.update(clause_files)
            if not clause_files and re.search(r"\b(?:it|its|them|their)\b", words):
                excluded.update(recent_refs or case.files)
            if not clause_files and _OWN_REFERENCE.search(words):
                doc, video = bool(_DOCUMENT.search(words)), bool(_VIDEO.search(words))
                kind = "document" if doc and not video else "video" if video and not doc else None
                excluded.update(file for file in case.files if kind is None or file.kind == kind)
            continue
        for token in _FILE_TOKEN.finditer(clause):
            if not any(
                start <= token.start() and token.end() <= end for _, (start, end) in matches
            ):
                missing_target = True
        if not negated and not management and not metadata and not transform and not meta_task:
            affirmative.append(words)
            if _INTENT.search(words):
                affirmative_refs.extend(clause_files)
            affirmative_evidence |= content_request or bool(
                _INTENT.search(words) and (clause_files or _OWN_REFERENCE.search(words))
            )

    if missing_target:
        return None
    intent_text = " ".join(affirmative)
    # Generic explanations after a meta-task are not permission to read its
    # illustrative filename. A scoped method switch can refer to real evidence
    # by pronoun, such as "do not summarize the file; compare its sections".
    if (
        not has_meta_task
        and not has_transform
        and any(file not in excluded for file in references)
        and _CONTENT_ACTION.search(intent_text)
        and re.search(r"\b(?:it|its|them|their)\b", intent_text, re.I)
    ):
        affirmative_evidence = True
    if not affirmative_evidence and has_transform:
        return None
    if not affirmative_evidence and has_meta_task:
        return None
    if not affirmative_evidence and has_management:
        return None
    if not affirmative_evidence and has_metadata_task:
        return None
    if not affirmative_evidence and negative_analysis:
        return None

    if not affirmative_evidence:
        return None

    # Named affirmative targets take precedence over names in prohibitions.
    targets = list(dict.fromkeys(affirmative_refs))
    if not targets:
        targets = [file for file in dict.fromkeys(references) if file not in excluded]
    if any(file in excluded for file in targets):
        return None
    if not targets and references and not affirmative_evidence:
        return None
    if not targets and _OWN_REFERENCE.search(intent_text):
        doc = bool(_DOCUMENT.search(intent_text))
        video = bool(_VIDEO.search(intent_text))
        kind = "document" if doc and not video else "video" if video and not doc else None
        targets = [
            file
            for file in case.files
            if file not in excluded and (kind is None or file.kind == kind)
        ]
    if not targets:
        return None
    if any(file.status != "ready" for file in targets):
        return None
    kinds = {file.kind for file in targets}
    if len(kinds) != 1:
        return None
    if not kinds <= {"document", "video"}:
        return None
    if not _INTENT.search(intent_text):
        return None
    return "grounded-document-analysis" if kinds == {"document"} else "video-analysis"


def select_automatic_skill(prompt: str, files: tuple[SkillEvidence, ...]) -> str | None:
    """Choose at most one existing workflow from bounded request metadata."""

    # Oversized requests/manifests retain the normal pipeline. This bounds regex
    # work independently of the model's normal request/context limits.
    if not files or len(files) > 100 or len(prompt) > 20_000:
        return None
    if any(not file.filename or len(file.filename) > 300 for file in files):
        return None
    # Equal names cannot reliably identify different file kinds/readiness.
    names = {}
    for file in files:
        key = file.filename.casefold()
        if key in names and names[key] != file:
            return None
        names[key] = file
    return _select(_Request(prompt=prompt, files=tuple(names.values())))
