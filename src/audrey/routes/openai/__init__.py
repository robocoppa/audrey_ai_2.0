"""OpenAI-compatible routes (package).

Exposes the virtual models declared by VIRTUAL_MODELS plus
/v1/chat/completions and /v1/responses. /v1/models is the authoritative
runtime inventory.

Plus an opt-in passthrough family selected via a model-string prefix:
  audrey_passthrough/<concrete>  — forward straight to Ollama, no
                                   classifier, no banners. Both fair-
                                   scheduling layers still fire so the
                                   request shares the GPU under the
                                   same rules as pipeline traffic.

Pipeline requests go through:
  classify → complexity gate → fast path | deep panel + synth.

Supported Chat Completions and Responses requests keep their respective
OpenAI wire shapes while sharing Audrey's generation path.

This was a single 1497-line module; it's now a package split by
responsibility (`schemas`, `responses`, `passthrough`, `pipeline`,
`routes`). The public surface re-exported here — `router`, `VIRTUAL_MODELS`,
the request schemas, `list_models`, and the passthrough helpers the tests
reach for — keeps `from audrey.routes.openai import …` unchanged for every
consumer (`main.py` and the test suite).
"""

from __future__ import annotations

from audrey.routes.openai.passthrough import (
    PASSTHROUGH_BARE,
    PASSTHROUGH_PREFIX,
    _handle_passthrough,
    _is_passthrough,
    _passthrough_concrete,
    _resolve_passthrough_model,
)
from audrey.routes.openai.routes import VIRTUAL_MODELS, create_response, list_models, router
from audrey.routes.openai.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ResponseCreateRequest,
    ResponseInputMessage,
)

__all__ = [
    "router",
    "VIRTUAL_MODELS",
    "list_models",
    "ChatMessage",
    "ChatCompletionRequest",
    "ResponseCreateRequest",
    "ResponseInputMessage",
    "create_response",
    "PASSTHROUGH_PREFIX",
    "PASSTHROUGH_BARE",
    "_is_passthrough",
    "_passthrough_concrete",
    "_resolve_passthrough_model",
    "_handle_passthrough",
]
