"""Classification — decides which task family a prompt belongs to.

Two-stage:
  1. **Keyword pre-filter** — cheap regex match on the last user message.
     Strong signals (e.g. triple-backtick code block, `def foo(`) pick a
     task directly. Weak signals are held in reserve as a router-failure
     fallback — they do not influence the router call itself.
  2. **Router model** — `qwen3:4b` produces a JSON verdict. On two strikes
     (timeout, parse error, unknown label) we fall back to the best keyword
     signal (strong or weak), or "general" if nothing matched.

**Review override:** "review this code" / "analyze this snippet" → reasoning,
even though the message contains code. Reviewing code is an analytical task,
not a code-generation one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from audrey.models.ollama import OllamaClient, OllamaError
from audrey.pipeline.prompts import CLASSIFIER_SYSTEM, prompt_from_config
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
    r"\b(bug|stack ?trace|traceback|exception|compile|syntax error|"
    r"python|javascript|typescript|rust|golang|kotlin|dockerfile|npm|pip install|uv add|"
    r"cargo|gradle|make|cmake|"
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


def _tool_mention_signal(text: str, tool_names: set[str]) -> KeywordSignal | None:
    """If the prompt explicitly names a registered tool, classify as `general`.

    A prompt like "use kb_image_search to find …" would otherwise trip
    `_VL_STRONG` on the word "image" and route to a tool-blind vl model.
    Naming a tool is a stronger signal than the surrounding nouns: the
    user wants tool dispatch, so route through the tool-capable fast path.
    """
    if not tool_names:
        return None
    lowered = text.lower()
    for name in tool_names:
        # Word boundary so "memory" alone doesn't trigger "memory_store".
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return KeywordSignal("general", "strong", f"tool_mention:{name}")
    return None


def keyword_classify(text: str, *, tool_names: set[str] | None = None) -> KeywordSignal | None:
    """Return the strongest keyword signal, or None if nothing matches."""
    # Tool-mention wins over everything else — the user is asking for
    # tool dispatch, which the deep / vl pools can't do in this phase.
    if tool_names:
        sig = _tool_mention_signal(text, tool_names)
        if sig is not None:
            return sig

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

# Prompt centralized in pipeline/prompts.py. `_ROUTER_SYSTEM` is kept as
# a local alias so the existing call site reads naturally; the source of
# truth is `prompts.CLASSIFIER_SYSTEM`.
_ROUTER_SYSTEM = CLASSIFIER_SYSTEM

_VALID_TASKS: set[TaskType] = {"code", "reasoning", "general", "vl"}


async def router_classify(
    ollama: OllamaClient,
    *,
    router_model: str,
    user_text: str,
    timeout_s: float,
    cfg: Any = None,
) -> tuple[TaskType | None, float, str]:
    """Ask the router model. Returns (task | None, confidence, raw_body_or_error).

    `cfg` is optional. When supplied, `agentic.prompts.classifier` overrides
    the default system prompt; missing/empty falls back to `_ROUTER_SYSTEM`.
    """
    system_prompt = prompt_from_config(cfg, "classifier", _ROUTER_SYSTEM)
    messages = [
        {"role": "system", "content": system_prompt},
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
    tool_names: set[str] | None = None,
    cfg: Any = None,
) -> tuple[TaskType, str, float]:
    """Classify with keyword short-circuit + router fallback.

    Returns `(task_type, reason, confidence)`.

    Decision order:
      1. Strong keyword signal → use it immediately.
      2. Run router up to `max_router_strikes` times.
      3. If router still failed, use weak-keyword signal if any.
      4. Default: "general", confidence 0.25.

    `tool_names`: the set of registered tool names (e.g. {"kb_search",
    "web_search", ...}). When the user prompt explicitly names one,
    we classify as `general` so the tool-capable fast path runs.
    """
    signal = keyword_classify(user_text, tool_names=tool_names)
    if signal is not None and signal.strength == "strong":
        return signal.task, f"keyword:{signal.reason}", 0.95

    strikes = 0
    last_err = ""
    while strikes < max_router_strikes:
        task, conf, info = await router_classify(
            ollama, router_model=router_model, user_text=user_text,
            timeout_s=router_timeout_s, cfg=cfg,
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


async def classify_with_registry(
    ollama: OllamaClient,
    *,
    user_text: str,
    router_cfg: dict[str, Any],
    cfg: Any = None,
    registry: Any = None,
) -> tuple[TaskType, str, float]:
    """Run `classify(...)` with the current tool-registry names threaded in.

    Both the non-streaming graph node and the streaming route need the same
    setup: read `router.*` config, extract `tool_names` from the live tool
    registry, call `classify(...)`. Inlining this in two places once silently
    diverged — the streaming path forgot the `tool_names` argument and lost
    the explicit-tool-mention routing override.

    `registry` is duck-typed against `ToolRegistry`: only `.names()` is
    used. `None` is fine — that path just produces an empty tool-name set,
    same as the previous inline code.
    """
    tool_names = set(registry.names()) if registry is not None else set()
    return await classify(
        ollama,
        router_model=router_cfg.get("model", "qwen3:4b"),
        router_timeout_s=float(router_cfg.get("timeout_s", 20)),
        max_router_strikes=int(router_cfg.get("max_failures_before_fallback", 2)),
        user_text=user_text,
        tool_names=tool_names,
        cfg=cfg,
    )


__all__ = ["classify", "keyword_classify", "router_classify", "classify_with_registry"]
