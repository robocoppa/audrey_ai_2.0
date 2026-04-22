"""Classification — decides which task family a prompt belongs to.

Two-stage:
  1. **Keyword pre-filter** — cheap regex match on the last user message.
     Strong signals (e.g. triple-backtick code block, `def foo(`) pick a
     task directly. Weak signals just bump priority for stage 2.
  2. **Router model** — `qwen3:4b` produces a JSON verdict. On two strikes
     (timeout, parse error, unknown label) we fall back to the best keyword
     signal, or "general" if nothing matched.

**Review override:** "review this code" / "analyze this snippet" → reasoning,
even though the message contains code. Reviewing code is an analytical task,
not a code-generation one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.pipeline.state import TaskType

log = logging.getLogger(__name__)

# ─── Keyword signals ──────────────────────────────────────────────────

_CODE_STRONG = re.compile(
    r"```[a-zA-Z0-9_+-]*\n"                       # fenced code block
    r"|^\s*(def|class)\s+\w+\s*\("                # python def/class
    r"|^\s*(public|private|protected)\s+\w"       # java/C# modifiers
    r"|^\s*(func|fn)\s+\w+\s*\("                  # go/rust
    r"|^\s*(const|let|var|function)\s+\w+\s*=",   # js
    re.MULTILINE,
)
_CODE_WEAK = re.compile(
    r"\b(bug|stack ?trace|traceback|exception|compile|syntax error|typescript|"
    r"python|javascript|typescript|rust|golang|kotlin|dockerfile|npm|pip install|uv add|"
    r"pytest|unittest|mypy|ruff|eslint)\b",
    re.IGNORECASE,
)
_REASONING_STRONG = re.compile(
    r"\b(analy[sz]e|analysis|review|critique|evaluate|compare|tradeoff|"
    r"why (does|did|would|should)|explain (the|how|why)|prove|justify|reason(?:ing)?|"
    r"pros and cons)\b",
    re.IGNORECASE,
)
_VL_STRONG = re.compile(
    r"\b(image|photo|picture|screenshot|png|jpe?g|this rock|identify .* rock|"
    r"what (type|kind) of|what do you see)\b",
    re.IGNORECASE,
)

# "review this code" / "analyze this snippet" / "look at this function"
_REVIEW_OVERRIDE = re.compile(
    r"\b(review|analy[sz]e|look at|inspect|critique|audit|find (bugs|issues|problems))\b.*?"
    r"\b(code|snippet|function|class|method|script|implementation)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True, frozen=True)
class KeywordSignal:
    task: TaskType
    strength: str  # "strong" | "weak"
    reason: str


def keyword_classify(text: str) -> KeywordSignal | None:
    """Return the strongest keyword signal, or None if nothing matches."""
    # Review-override wins unconditionally — reviewing code is reasoning.
    if _REVIEW_OVERRIDE.search(text):
        return KeywordSignal("reasoning", "strong", "review_override")

    if _VL_STRONG.search(text):
        return KeywordSignal("vl", "strong", "vl_strong")
    if _CODE_STRONG.search(text):
        return KeywordSignal("code", "strong", "code_strong")
    if _REASONING_STRONG.search(text):
        return KeywordSignal("reasoning", "strong", "reasoning_strong")
    if _CODE_WEAK.search(text):
        return KeywordSignal("code", "weak", "code_weak")
    return None


# ─── Router model ─────────────────────────────────────────────────────

_ROUTER_SYSTEM = (
    "You are a task classifier. Read the user's message and output a JSON object "
    "with exactly these keys:\n"
    '  {"task": "code|reasoning|general|vl", "confidence": 0.0-1.0}\n'
    "Rules:\n"
    "- 'code' = user wants code written, debugged, refactored, explained line-by-line.\n"
    "- 'reasoning' = analysis, comparison, review, multi-step logic, math proofs, explanations.\n"
    "- 'vl' = anything referencing an image, photo, screenshot, or visual identification.\n"
    "- 'general' = chitchat, facts, summaries, everything else.\n"
    "Output ONLY the JSON object. No prose, no markdown."
)

_VALID_TASKS: set[TaskType] = {"code", "reasoning", "general", "vl"}


async def router_classify(
    ollama: OllamaClient,
    *,
    router_model: str,
    user_text: str,
    timeout_s: float,
) -> tuple[TaskType | None, float, str]:
    """Ask the router model. Returns (task | None, confidence, raw_body_or_error)."""
    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": user_text[:2000]},  # hard cap; routing is cheap
    ]
    try:
        resp = await ollama.chat(
            model=router_model,
            messages=messages,
            options={"temperature": 0.0},
            timeout_s=timeout_s,
        )
    except OllamaError as e:
        return None, 0.0, f"ollama_error:{e}"

    body = (resp.get("message", {}) or {}).get("content", "") or ""
    task, conf = _parse_router_output(body)
    if task is None:
        return None, 0.0, f"parse_error:{body[:200]}"
    return task, conf, body


def _parse_router_output(raw: str) -> tuple[TaskType | None, float]:
    raw = raw.strip()
    # Tolerate fences / surrounding prose — extract the first {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, 0.0
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None, 0.0
    task_raw = str(obj.get("task", "")).strip().lower()
    if task_raw not in _VALID_TASKS:
        return None, 0.0
    conf = obj.get("confidence", 0.5)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5
    return task_raw, conf  # type: ignore[return-value]


# ─── Top-level classify() ─────────────────────────────────────────────

async def classify(
    ollama: OllamaClient,
    *,
    router_model: str,
    router_timeout_s: float,
    max_router_strikes: int,
    user_text: str,
) -> tuple[TaskType, str, float]:
    """Classify with keyword short-circuit + router fallback.

    Returns `(task_type, reason, confidence)`.

    Decision order:
      1. Strong keyword signal → use it immediately.
      2. Run router up to `max_router_strikes` times.
      3. If router still failed, use weak-keyword signal if any.
      4. Default: "general", confidence 0.25.
    """
    signal = keyword_classify(user_text)
    if signal is not None and signal.strength == "strong":
        return signal.task, f"keyword:{signal.reason}", 0.95

    strikes = 0
    last_err = ""
    while strikes < max_router_strikes:
        task, conf, info = await router_classify(
            ollama, router_model=router_model, user_text=user_text, timeout_s=router_timeout_s
        )
        if task is not None:
            return task, f"router:{task}", conf
        strikes += 1
        last_err = info
        log.warning("router classify strike %d/%d: %s", strikes, max_router_strikes, info[:120])

    if signal is not None:
        return signal.task, f"fallback_keyword:{signal.reason}", 0.6

    log.warning("classify fell all the way through; defaulting to general. last=%s", last_err[:120])
    return "general", "fallback:general", 0.25


__all__ = ["classify", "keyword_classify", "router_classify"]
