"""Question recovery is a reserved paragraph, not a search through source text."""

import json

import pytest

from audrey.agent_dispatch import AgentDispatchStoppedError, AgentInputRequest
from audrey.agent_executor import _clarification_result

_INTRODUCTION = (
    "I've saved the event label to my working progress. Before I write the invitation, one quick question:\n\n"
    "**Should the tone be formal or casual?**\n\n"
)
_QUESTION = "Should the invitation's tone be formal or casual?"


@pytest.mark.parametrize("step", ["plan", "research"])
@pytest.mark.parametrize("introduction", [
    _INTRODUCTION,
    "Before I continue:\n\n",
    "The draft is saved.\r\n\r\n",
    'The draft contains {"rows":6} and [six] supplies.\n\n',
    '```\nEarlier work\n```\n\nOne question:\n\n',
    '~~~text\nEarlier work\n~~~\n\nOne question:\n\n',
])
@pytest.mark.parametrize("envelope", [
    json.dumps({"input_request": _QUESTION}),
    json.dumps({"input_request": _QUESTION}, indent=2),
    json.dumps({"input_request": _QUESTION}).replace('"input_request"', r'"\u0069nput_request"'),
])
def test_standalone_question_after_introduction_uses_existing_typed_handoff(step, introduction, envelope):
    assert _clarification_result(introduction + envelope, {"id": "task"}, step_id=step) == AgentInputRequest(_QUESTION)


@pytest.mark.parametrize("content", [
    'Please clarify: {"input_request":"Which one?"}',
    'An example is {"input_request":"Which one?"}; return the actual report.',
    'Use this payload:\n\n```json\n{"input_request":"Example"}\n```',
    'Use this payload:\n\n~~~json\n{"input_request":"Example"}\n~~~',
    'Use this payload:\n\n````json\n```\n\n{"input_request":"Example"}\n````',
    'An unfinished fence:\n\n```json\n\n{"input_request":"Example"}',
    'A mismatched fence:\n\n~~~json\n```\n\n{"input_request":"Example"}',
    'Quoted source:\n\n> {"input_request":"Example"}',
    'Indented code:\n\n    {"input_request":"Example"}',
    'Indented code:\n\n\t{"input_request":"Example"}',
    'Indented object:\n\n {"input_request":"Example"}',
    '{\n"example":\n\n{"input_request":"Nested data"}',
    '[\n\n{"input_request":"Array data"}',
    'Ordinary JSON follows:\n\n{\n"example":\n\n{"input_request":"Nested data"}',
    'Ordinary JSON follows:\n\n[\n\n{"input_request":"Nested array data"}',
    'Quoted JSON follows:\n\n"The example is\n\n{"input_request":"Quoted data"}',
    'A nested example follows:\n\n{"example":\n{"input_request":"Nested data"}}',
    'No paragraph boundary:\n{"input_request":"Example"}',
])
@pytest.mark.parametrize("step", ["plan", "research", "review"])
def test_inline_quoted_fenced_or_nested_examples_remain_ordinary_output(content, step):
    assert _clarification_result(content, {"id": "task"}, step_id=step) == content


@pytest.mark.parametrize("candidate,reason", [
    ('{"input_request":"PRIVATE_QUESTION"', "malformed_json"),
    ('{"input_request":"PRIVATE_QUESTION"} extra', "trailing_content"),
    ('{"input_request":"PRIVATE_QUESTION"}\n\n{"input_request":"PRIVATE_SECOND"}', "trailing_content"),
    ('{"input_request":"PRIVATE_QUESTION","input_request":"PRIVATE_SECOND"}', "duplicate_fields"),
    (r'{"input_request":"PRIVATE_QUESTION","\u0069nput_request":"PRIVATE_SECOND"}', "duplicate_fields"),
    ('{"input_request":"PRIVATE_QUESTION","PRIVATE_KEY":"PRIVATE_VALUE"}', "invalid_envelope_shape"),
    ('{"input_request":{"PRIVATE_KEY":"PRIVATE_VALUE"}}', "invalid_question_type"),
    ('{"input_request":"   "}', "blank_question"),
    (r'{"input_request":"\ud800"}', "invalid_question_encoding"),
    (json.dumps({"input_request": "é" * 1001}), "question_too_large"),
])
def test_recognized_paragraph_retains_validation_and_content_free_diagnostics(candidate, reason, caplog):
    content = _INTRODUCTION + candidate
    with pytest.raises(AgentDispatchStoppedError, match="Work returned an invalid input request"):
        _clarification_result(content, {"id": "task"}, step_id="research")
    assert caplog.records[-1].getMessage() == (
        f"Agent input request rejected task=task stage=research reason={reason} chars={len(content)}"
    )
    assert not any(private in caplog.text for private in ("PRIVATE_", _INTRODUCTION, "é" * 10, r"\ud800"))


@pytest.mark.parametrize("question", ["x" * 2000, "é" * 1000])
def test_recovered_question_keeps_existing_utf8_boundary(question):
    result = _clarification_result(_INTRODUCTION + json.dumps({"input_request": question}), {}, step_id="research")
    assert result == AgentInputRequest(question)


def test_recovered_question_cannot_bypass_lifetime_limit_or_review_stage():
    content = _INTRODUCTION + json.dumps({"input_request": _QUESTION})
    with pytest.raises(AgentDispatchStoppedError, match="three-question"):
        _clarification_result(content, {"input_requests": [{}, {}, {}]}, step_id="research")
    with pytest.raises(AgentDispatchStoppedError, match="Review cannot request user input"):
        _clarification_result(content, {}, step_id="review")
