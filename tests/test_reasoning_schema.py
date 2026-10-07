"""Reasoning controls must not silently vanish at the request boundary."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from audrey.routes.openai.schemas import ChatCompletionRequest, ResponseCreateRequest


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", None])
def test_reasoning_vocabulary_is_retained_for_model_admission(effort):
    chat = ChatCompletionRequest(model="audrey_passthrough/model", messages=[{"role": "user", "content": "Hi"}],
                                 reasoning_effort=effort)
    response = ResponseCreateRequest(model="audrey_passthrough/model", input="Hi", reasoning={"effort": effort})
    assert chat.reasoning_effort == effort
    assert response.reasoning.effort == effort


@pytest.mark.parametrize("effort", [True, False, 1, 0, "HIGH", "enabled", {}, []])
def test_malformed_reasoning_is_rejected_in_both_protocols(effort):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="audrey_passthrough/model", messages=[{"role": "user", "content": "Hi"}],
                              reasoning_effort=effort)
    with pytest.raises(ValidationError):
        ResponseCreateRequest(model="audrey_passthrough/model", input="Hi", reasoning={"effort": effort})


@pytest.mark.parametrize("config", [{"effort": "low", "summary": "auto"}, {"mode": "pro"}, "low", True])
def test_responses_does_not_accept_other_reasoning_features(config):
    with pytest.raises(ValidationError):
        ResponseCreateRequest(model="audrey_passthrough/model", input="Hi", reasoning=config)


@pytest.mark.parametrize("config", [None, {}, {"effort": None}])
def test_empty_reasoning_does_not_create_an_effort_override(config):
    response = ResponseCreateRequest(model="audrey_fast", input="Hi", reasoning=config)
    assert response.reasoning is None or response.reasoning.effort is None
