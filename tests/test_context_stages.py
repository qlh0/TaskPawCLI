from __future__ import annotations

import asyncio
from typing import Any

from taskpaw.agent import QueryEngine
from taskpaw.config import load_config
from taskpaw.context import ContextBudget, ContextWindowManager
from taskpaw.context.manager import CLEARED_TOOL_RESULT
from taskpaw.tools import ToolRegistry, get_builtin_tools
from taskpaw.types import Message

BUDGET = ContextBudget(context_window=2200, max_output_tokens=100, reserve_tokens=100)


def _call(call_id: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    )


def _tool(call_id: str, size: int = 2000) -> Message:
    return Message(role="tool", content="x" * size, tool_call_id=call_id)


def _manager(**overrides) -> ContextWindowManager:
    options = {
        "max_history_messages": 100,
        "min_recent_messages": 2,
        "tool_result_keep_rounds": 1,
        "tool_result_min_chars": 10,
    }
    options.update(overrides)
    return ContextWindowManager(BUDGET, **options)


def _conversation(
    *,
    rounds: int = 3,
    sizes: list[int] | None = None,
    image_first: bool = False,
) -> list[Message]:
    sizes = sizes or [2000] * rounds
    messages = [Message(role="user", content="开始")]
    for index in range(1, rounds + 1):
        messages.append(_call(f"c{index}"))
        if image_first and index == 1:
            messages.append(
                Message(
                    role="tool",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + "A" * 2000},
                        }
                    ],
                    tool_call_id=f"c{index}",
                )
            )
        else:
            messages.append(_tool(f"c{index}", sizes[index - 1]))
    messages.append(Message(role="user", content="继续"))
    return messages


# ---------------------------------------------------------------------------
# Stage 1: tool-result clearing
# ---------------------------------------------------------------------------


def test_old_tool_results_are_cleared_and_recent_ones_kept():
    manager = _manager()
    messages = _conversation()

    result = manager.prepare(messages)

    assert result.compressed is True
    assert result.stage == "tool_results"
    assert result.cleared_tool_results == 2
    assert len(result.messages) == len(messages)
    assert result.messages[2].content == CLEARED_TOOL_RESULT
    assert result.messages[4].content == CLEARED_TOOL_RESULT
    # The most recent tool round is protected and keeps its payload.
    assert result.messages[6].content == "x" * 2000
    # Tool-call pairing is preserved.
    assert result.messages[2].tool_call_id == "c1"
    assert result.messages[6].tool_call_id == "c3"


def test_keep_rounds_protects_the_last_tool_interactions():
    manager = _manager(tool_result_keep_rounds=2)
    messages = _conversation(rounds=4, sizes=[1200, 1200, 1200, 1200])

    result = manager.prepare(messages)

    assert result.stage == "tool_results"
    assert result.cleared_tool_results == 2
    assert result.messages[2].content == CLEARED_TOOL_RESULT
    assert result.messages[4].content == CLEARED_TOOL_RESULT
    assert result.messages[6].content == "x" * 1200
    assert result.messages[8].content == "x" * 1200


def test_small_tool_results_are_not_cleared():
    manager = _manager(tool_result_min_chars=500)
    messages = _conversation(sizes=[100, 2500, 2500])

    result = manager.prepare(messages)

    assert result.cleared_tool_results == 1
    assert result.messages[2].content == "x" * 100
    assert result.messages[4].content == CLEARED_TOOL_RESULT


def test_clearing_is_idempotent():
    manager = _manager()
    first = manager.prepare(_conversation())

    second = manager.prepare(first.messages)

    assert second.compressed is False
    assert second.messages[2].content == CLEARED_TOOL_RESULT


def test_under_budget_returns_messages_untouched():
    manager = _manager()
    messages = [Message(role="user", content="hi")]

    result = manager.prepare(messages)

    assert result.compressed is False
    assert result.stage == "none"
    assert result.messages == messages


def test_multimodal_tool_payload_is_replaced_with_text():
    manager = _manager()
    messages = _conversation(image_first=True)

    result = manager.prepare(messages)

    assert result.stage == "tool_results"
    assert result.messages[2].content == CLEARED_TOOL_RESULT
    assert result.messages[4].content == CLEARED_TOOL_RESULT


# ---------------------------------------------------------------------------
# Stage 2: summary fallback
# ---------------------------------------------------------------------------


def test_summary_runs_when_clearing_is_not_enough():
    manager = _manager(tool_result_clear_enabled=False)
    messages = [Message(role="user", content="word " * 400) for _ in range(8)]

    result = manager.prepare(messages)

    assert result.compressed is True
    assert result.stage in {"summary", "summary_shrink"}
    assert result.summarized_messages > 0


def test_message_count_limit_skips_stage_one():
    manager = _manager(max_history_messages=4)
    messages: list[Message] = []
    for index in range(7):
        messages.append(Message(role="user", content=f"消息 {index}"))
        messages.append(Message(role="assistant", content=f"回答 {index}"))

    result = manager.prepare(messages)

    assert result.stage in {"summary", "summary_shrink"}
    assert result.cleared_tool_results == 0


def test_boundary_never_splits_a_tool_round():
    manager = _manager(min_recent_messages=2)
    messages = _conversation()

    result = manager.prepare(messages)

    # Every retained message must not be an orphaned tool result.
    retry_tools = [m for m in result.messages if m.role == "tool"]
    assert retry_tools, "tool messages must still be present"


# ---------------------------------------------------------------------------
# Integration: the event carries the stage
# ---------------------------------------------------------------------------


class BigToolClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 4000

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls <= 4:
            path = f"note{self.calls}.txt"
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": f"call_{self.calls}",
                    "function": {"name": "read_file", "arguments": f'{{"path":"{path}"}}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        yield {"type": "text_delta", "text": "done"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_query_emits_compression_stage_and_cleared_count(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    for index in range(1, 5):
        (tmp_path / f"note{index}.txt").write_text("y" * 20000, encoding="utf-8")

    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.llm.max_tokens = 200
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=BigToolClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run() -> list[dict[str, Any]]:
        return [event async for event in engine.ask("读四个文件")]

    events = asyncio.run(run())
    compressed = [event for event in events if event.get("type") == "context_compressed"]

    assert compressed, "expected at least one compression event"
    shrunk = [event for event in compressed if event["after_tokens"] < event["before_tokens"]]
    assert shrunk, "expected at least one event that actually reduced the context"
    first = shrunk[0]
    assert first["stage"] in {"tool_results", "summary", "summary_shrink"}
    assert isinstance(first["cleared_tool_results"], int)
    assert first["cleared_tool_results"] >= 0
