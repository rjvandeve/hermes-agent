"""Truncated tool-call recovery must change preconditions (chunk/checkpoint guidance),
never execute incomplete arguments, and still refuse after the bounded retry cap.

Replays the recorded production path:
  Truncated tool call detected — retrying API call (N/4)...
  Truncated tool call response detected again — refusing to execute incomplete tool arguments
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_truncation import (
    _TOOL_CALL_TRUNCATION_CHUNK_NUDGE,
    _TOOL_CALL_TRUNCATION_NUDGE_FLAG,
    _retry_truncated_tool_call,
    inject_tool_call_truncation_guidance,
    recover_from_truncation,
    tool_call_truncation_guidance,
)
from agent.turn_retry_state import TurnRetryState
from hermes_constants import PARTIAL_STREAM_STUB_ID


def _agent(**extra):
    a = SimpleNamespace(
        max_tokens=4096,
        _ephemeral_max_output_tokens=None,
        _buffer_vprint=lambda *a, **k: None,
        _vprint=lambda *a, **k: None,
        _flush_status_buffer=lambda: None,
        _cleanup_task_resources=lambda *a, **k: None,
        _persist_session=lambda *a, **k: None,
        _requested_output_cap_from_api_kwargs=lambda kw: 4096,
        _session_messages=None,
        log_prefix="",
        provider="openrouter",
        api_mode="chat_completions",
        **extra,
    )
    return a


def _st(agent, *, retries=0, is_stub=False, messages=None, response=None):
    st = SimpleNamespace(
        agent=agent,
        truncated_tool_call_retries=retries,
        is_stub=is_stub,
        messages=list(messages or [{"role": "user", "content": "write the report"}]),
        response=response or SimpleNamespace(),
        effective_task_id="t1",
        api_call_count=1,
        conversation_history=[],
        length_continue_retries=0,
        truncated_response_parts=[],
        retry_count=0,
        compression_attempts=0,
        current_turn_user_idx=0,
        finish_reason="length",
        window_filled=None,
        action="fallthrough",
        result=None,
    )

    def done(action, result=None):
        st.action, st.result = action, result
        return st

    def end_turn(final_response, error=None, *, result_messages=None, cleanup=True,
                 failed=False, compression_exhausted=False, failure=("truncated", True)):
        st.action = "return"
        st.result = {
            "final_response": final_response,
            "completed": False,
            "partial": True,
            "failure_reason": failure[0],
            "messages": result_messages if result_messages is not None else st.messages,
        }
        return st

    st.done = done
    st.end_turn = end_turn
    return st


def test_guidance_forbids_same_oversized_retry_and_keeps_refusal():
    text = tool_call_truncation_guidance(is_stub=False).lower()
    assert "refused to execute incomplete" in text or "incomplete arguments" in text
    assert "do not retry the same oversized" in text
    assert "smaller" in text
    assert "checkpoint" in text


def test_inject_replaces_tail_nudge_instead_of_stacking():
    messages = [{"role": "user", "content": "task"}]
    inject_tool_call_truncation_guidance(messages)
    inject_tool_call_truncation_guidance(messages)
    inject_tool_call_truncation_guidance(messages)
    assert sum(1 for m in messages if m.get(_TOOL_CALL_TRUNCATION_NUDGE_FLAG)) == 1
    assert messages[-1]["content"] == _TOOL_CALL_TRUNCATION_CHUNK_NUDGE
    # original task user row preserved
    assert messages[0]["content"] == "task"


def test_retry_injects_guidance_and_boosts_without_appending_tool_calls():
    agent = _agent()
    st = _st(agent)
    verdict = _retry_truncated_tool_call(st, {})
    assert verdict.action == "continue"
    assert st.truncated_tool_call_retries == 1
    assert agent._ephemeral_max_output_tokens and agent._ephemeral_max_output_tokens > 4096
    # No assistant/tool_calls row — incomplete args never enter the transcript
    assert not any(m.get("tool_calls") for m in st.messages if isinstance(m, dict))
    assert st.messages[-1].get(_TOOL_CALL_TRUNCATION_NUDGE_FLAG) is True
    assert "Do NOT retry the same oversized" in st.messages[-1]["content"]
    assert agent._session_messages is st.messages


def test_four_retries_then_refuse_still_does_not_execute():
    agent = _agent()
    st = _st(agent)
    for i in range(4):
        v = _retry_truncated_tool_call(st, {})
        assert v.action == "continue", f"attempt {i+1}"
    # 5th path = refuse
    v = _retry_truncated_tool_call(st, {})
    assert v.action == "return"
    assert v.result["partial"] is True
    assert v.result["failure_reason"] == "truncated"
    # Still no tool_calls in messages
    assert not any(m.get("tool_calls") for m in st.messages if isinstance(m, dict))
    # Guidance present but not stacked 4 deep
    nudges = [m for m in st.messages if m.get(_TOOL_CALL_TRUNCATION_NUDGE_FLAG)]
    assert len(nudges) == 1


def test_stub_retry_uses_dropped_tools_continuation_copy():
    agent = _agent()
    resp = SimpleNamespace(id=PARTIAL_STREAM_STUB_ID, _dropped_tool_names=["write_file"])
    st = _st(agent, is_stub=True, response=resp)
    # is_stub property on real _Trunc uses response.id; emulate via flag already set
    _retry_truncated_tool_call(st, {})
    text = st.messages[-1]["content"].lower()
    assert "write_file" in text
    assert "smaller" in text
    assert "available" in text


def _mock_msg(*, content="", tool_calls=None):
    m = SimpleNamespace(content=content, tool_calls=tool_calls)
    return m


def test_recover_from_truncation_tool_path_injects_guidance(monkeypatch):
    """End-to-end through recover_from_truncation (the production wrapper entry)."""
    agent = _agent()
    agent._get_transport = lambda: SimpleNamespace(
        normalize_response=lambda r, **k: _mock_msg(
            tool_calls=[SimpleNamespace(function=SimpleNamespace(name="write_file", arguments='{"x":'))],
        )
    )
    agent._strip_think_blocks = lambda c: c or ""
    agent._has_content_after_think_block = lambda c: True

    bad = SimpleNamespace(
        id="resp-1",
        usage=None,
        tool_calls=[SimpleNamespace()],
        choices=[SimpleNamespace(message=_mock_msg(tool_calls=[SimpleNamespace()]))],
    )
    # normalize_response_for_agent uses transport — already patched via _get_transport

    messages = [{"role": "user", "content": "write big file"}]
    verdict = recover_from_truncation(
        agent, bad, "length", TurnRetryState(),
        messages=messages, conversation_history=[], api_kwargs={},
        api_call_count=1, effective_task_id="t1", current_turn_user_idx=0,
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )
    assert verdict.action == "continue"
    assert any(m.get(_TOOL_CALL_TRUNCATION_NUDGE_FLAG) for m in verdict.messages)
    assert verdict.truncated_tool_call_retries == 1


def test_mutation_without_inject_is_same_input_retry():
    """Falsification: old behavior (boost only, no guidance) leaves messages unchanged."""
    agent = _agent()
    messages = [{"role": "user", "content": "task"}]
    st = _st(agent, messages=messages)
    # Simulate pre-fix path: boost only
    st.truncated_tool_call_retries = 0
    st.truncated_tool_call_retries += 1
    agent._ephemeral_max_output_tokens = 8192
    # messages unchanged → same-input retry (the audited defect)
    assert st.messages == [{"role": "user", "content": "task"}]
    # Fixed path changes messages
    inject_tool_call_truncation_guidance(st.messages)
    assert st.messages != [{"role": "user", "content": "task"}]
    assert "Do NOT retry the same oversized" in st.messages[-1]["content"]
