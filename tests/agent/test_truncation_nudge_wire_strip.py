"""Private truncation-recovery markers stay in history, never on a provider request copy.

``inject_tool_call_truncation_guidance`` tags its user row ``_tool_call_truncation_nudge``
(+ the older ``_length_continuation_nudge``) so retries replace instead of stacking. Some
transports keep unknown underscore keys, so ``build_api_messages`` — the copy every
transport receives — must drop both, and the token estimator must not price them.
"""
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.message_metadata import NON_WIRE_MESSAGE_FIELDS
from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import build_api_messages
from agent.turn_truncation import (
    _TOOL_CALL_TRUNCATION_CHUNK_NUDGE,
    _TOOL_CALL_TRUNCATION_NUDGE_FLAG,
    inject_tool_call_truncation_guidance,
)
from run_agent import AIAgent

PRIVATE = (_TOOL_CALL_TRUNCATION_NUDGE_FLAG, "_length_continuation_nudge", "_length_continuation_fragment")
SYSTEM = "You are helpful."


@pytest.fixture(autouse=True)
def _no_plugin_discovery(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


@pytest.fixture()
def agent():
    tool_defs = [{"type": "function", "function": {
        "name": "write_file", "description": "write", "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    a.client = MagicMock()
    a._cached_system_prompt = SYSTEM
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    a.valid_tool_names.add("write_file")
    return a


def _build(agent, messages, current_idx):
    agent._current_turn_timestamp = time.time()
    return build_api_messages(
        agent, messages, current_turn_user_idx=current_idx, ext_prefetch_cache=None,
        plugin_user_context=None, moa_config=None, active_system_prompt=SYSTEM,
    )


def _leaked(api_messages):
    return [(i, k) for i, m in enumerate(api_messages) if isinstance(m, dict) for k in m if k in PRIVATE]


def test_builder_strips_new_and_old_markers_but_history_keeps_them(agent):
    messages = [
        {"role": "user", "content": "earlier", "api_content": "earlier"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "write the report", "api_content": "write the report"},
    ]
    before, sys_before = _build(agent, messages, 2)

    inject_tool_call_truncation_guidance(messages)  # the real producer
    after, sys_after = _build(agent, messages, 2)

    # Private recovery state is retained internally (retry de-dupe reads it).
    assert messages[-1][_TOOL_CALL_TRUNCATION_NUDGE_FLAG] is True
    assert messages[-1]["_length_continuation_nudge"] is True
    # ...and never reaches the request copy (new-marker negative + old-marker control).
    assert _leaked(after) == []
    assert after[-1] == {"role": "user", "content": _TOOL_CALL_TRUNCATION_CHUNK_NUDGE}
    # Prior rows and the system prefix are byte-identical: cache prefix unchanged.
    assert sys_after == sys_before == SYSTEM
    assert json.dumps(after[: len(before)], sort_keys=True) == json.dumps(before, sort_keys=True)


def test_old_marker_alone_still_stripped_positive_control(agent):
    messages = [
        {"role": "user", "content": "task", "api_content": "task"},
        {"role": "assistant", "content": "partial", "_length_continuation_fragment": True},
        {"role": "user", "content": "continue", "_length_continuation_nudge": True},
    ]
    api_messages, _ = _build(agent, messages, 0)
    assert _leaked(api_messages) == []
    assert messages[2]["_length_continuation_nudge"] is True


def test_estimator_does_not_price_private_markers():
    tagged = [{"role": "user", "content": "x"}]
    inject_tool_call_truncation_guidance(tagged)
    stripped = [{k: v for k, v in m.items() if k not in NON_WIRE_MESSAGE_FIELDS} for m in tagged]
    assert any(k in PRIVATE for m in tagged for k in m)
    assert estimate_messages_tokens_rough(tagged) == estimate_messages_tokens_rough(stripped)


def _resp(finish_reason, tool_args):
    tc = SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                         function=SimpleNamespace(name="write_file", arguments=tool_args))
    msg = SimpleNamespace(content="", tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           model="test/model", usage=None)


def _run_ladder(agent, responses):
    """Real run_conversation phase dispatch with the given provider responses.

    Captures the api_messages every transport receives (``_build_api_kwargs`` input,
    before chat-completions' own underscore sweep), the internal history at that moment,
    and the request ``max_tokens`` that ``truncated_tool_call_retries`` boosts.
    """
    seen, budgets, history = [], [], []
    real_build = agent._build_api_kwargs

    def spy(api_messages, *a, **k):
        seen.append(json.loads(json.dumps(api_messages, default=str)))
        hist = getattr(agent, "_session_messages", None) or []
        history.append([(m.get("role"), m.get("content"), bool(m.get(_TOOL_CALL_TRUNCATION_NUDGE_FLAG)))
                        for m in hist if isinstance(m, dict)])
        kwargs = real_build(api_messages, *a, **k)
        budgets.append(kwargs.get("max_tokens"))
        return kwargs

    with (
        patch("model_tools.handle_function_call", return_value='{"success": true}') as hfc,
        patch.object(agent, "_build_api_kwargs", side_effect=spy),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.client.chat.completions.create.side_effect = responses
        result = agent.run_conversation("write the report")
    assert len(seen) == len(responses), f"bounded: 1 + 4 retries, got {len(seen)}"
    assert result.get("partial") is True  # refused after the cap
    for n, api_messages in enumerate(seen):
        assert _leaked(api_messages) == [], f"attempt {n + 1} leaked {_leaked(api_messages)}"
    return seen, budgets, history, hfc


def _truncated():
    return _resp("length", '{"path":"r.md","content":"partial')


def _assert_budget_ladder(budgets):
    # Retry counter drives the output budget: unset on the first truncation, rising after.
    assert budgets[0] is None
    assert all(a < b for a, b in zip(budgets[1:], budgets[2:])), budgets


def test_ladder_after_tool_step_nudge_row_reaches_wire_without_marker(agent):
    """Multi-step shape: the nudge follows a tool result, so it is its own wire row."""
    seen, budgets, history, hfc = _run_ladder(
        agent, [_resp("tool_calls", '{"path":"a.md","content":"ok"}')] + [_truncated() for _ in range(5)],
    )
    hfc.assert_called_once()  # only the complete call ran; truncated args never execute
    _assert_budget_ladder(budgets[1:])
    prefix = seen[1]  # [system, request, assistant tool_call, tool result]: first truncation
    assert [m["role"] for m in prefix] == ["system", "user", "assistant", "tool"]
    for api_messages in seen[2:]:
        # Prior rows byte-identical (cache prefix); one guidance row, replaced not stacked.
        assert json.dumps(api_messages[: len(prefix)], sort_keys=True) == json.dumps(prefix, sort_keys=True)
        assert api_messages[len(prefix):] == [{"role": "user", "content": _TOOL_CALL_TRUNCATION_CHUNK_NUDGE}]
    for rows in history[2:]:  # history keeps the private tag on exactly one row
        assert [r for r in rows if r[2]] == [("user", _TOOL_CALL_TRUNCATION_CHUNK_NUDGE, True)]


def test_ladder_on_request_never_rewrites_user_row_or_stacks(agent):
    """Nudge directly after the user's request: history must keep two rows (the per-call
    copy merges them for alternation); folding in place rewrote the request and stacked."""
    seen, budgets, history, hfc = _run_ladder(agent, [_truncated() for _ in range(5)])
    hfc.assert_not_called()
    _assert_budget_ladder(budgets)
    task = ("user", "write the report", False)
    nudge = ("user", _TOOL_CALL_TRUNCATION_CHUNK_NUDGE, True)
    assert history[1:] == [[task, nudge]] * 4
    system, request = seen[0]
    assert request == {"role": "user", "content": "write the report"}
    for api_messages in seen[1:]:
        assert api_messages == [system, {"role": "user", "content": "write the report\n\n" + _TOOL_CALL_TRUNCATION_CHUNK_NUDGE}]
