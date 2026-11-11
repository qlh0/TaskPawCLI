from __future__ import annotations

import asyncio
from typing import Any

from taskpaw.agent import Agent, PlanExecuteAgent
from taskpaw.config import load_config
from taskpaw.plan import PlanDecision
from taskpaw.tools import ToolRegistry, get_builtin_tools

PLAN_A = '{"summary":"a","tasks":[{"id":"a","description":"任务 A","type":"ANALYSIS"}]}'
PLAN_B = '{"summary":"b","tasks":[{"id":"b","description":"任务 B","type":"ANALYSIS"}]}'


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class ApprovalClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self) -> None:
        self.plan_calls = 0
        self.task_messages: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            self.plan_calls += 1
            yield {"type": "text_delta", "text": PLAN_B if "修改意见" in body else PLAN_A}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        self.task_messages.append(body)
        yield {"type": "text_delta", "text": "任务完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _agent(tmp_path, monkeypatch, client, callback, *, goal_mode: str = "plan"):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.policy.hitl_mode = "never"
    config.plan.require_approval = True
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return PlanExecuteAgent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        plan_approval_callback=callback,
    )


def _run(agent, goal: str = "先执行任务 A，然后执行任务 B"):
    async def runner():
        text = ""
        events = []
        async for event in agent.run(goal):
            events.append(event)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text, events

    return asyncio.run(runner())


def _done(events: list[dict]) -> dict:
    return next(event for event in events if event.get("type") == "done")


def test_plan_is_approved_before_any_task_runs(tmp_path, monkeypatch):
    client = ApprovalClient()
    seen: list[str] = []
    agent = _agent(tmp_path, monkeypatch, client, lambda plan: seen.append(plan.id) or "approve")

    text, events = _run(agent)

    assert seen, "callback should receive the plan"
    assert any(event.get("type") == "plan_approved" for event in events)
    assert any(event.get("type") == "plan_ready" for event in events)
    assert "已完成 [task_1]" in text
    assert _done(events)["stop_reason"] == "end_turn"


def test_plan_can_be_cancelled_before_execution(tmp_path, monkeypatch):
    client = ApprovalClient()
    agent = _agent(tmp_path, monkeypatch, client, lambda plan: PlanDecision("cancel"))

    text, events = _run(agent)

    assert any(event.get("type") == "plan_cancelled" for event in events)
    assert not any(event.get("type") == "plan_task_started" for event in events)
    assert client.task_messages == []
    assert "已取消计划执行" in text
    assert _done(events)["stop_reason"] == "plan_cancelled"


def test_revise_replans_before_execution(tmp_path, monkeypatch):
    client = ApprovalClient()
    decisions = [PlanDecision("revise", "第二个任务改成任务 B"), PlanDecision("approve")]
    calls: list[int] = []

    def callback(plan):
        calls.append(1)
        return decisions[len(calls) - 1]

    agent = _agent(tmp_path, monkeypatch, client, callback)
    text, events = _run(agent)

    assert len(calls) == 2
    assert client.plan_calls == 2
    assert any(event.get("type") == "plan_revision_started" for event in events)
    assert any("任务 B" in message for message in client.task_messages)
    assert "已完成 [task_1]" in text
    assert _done(events)["stop_reason"] == "end_turn"


def test_async_approval_callback_is_supported(tmp_path, monkeypatch):
    client = ApprovalClient()

    async def callback(plan):  # noqa: ARG001
        await asyncio.sleep(0)
        return "approve"

    agent = _agent(tmp_path, monkeypatch, client, callback)
    _, events = _run(agent)

    assert any(event.get("type") == "plan_approved" for event in events)
    assert _done(events)["stop_reason"] == "end_turn"


def test_agent_plan_mode_forwards_the_approval_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.policy.hitl_mode = "never"
    config.plan.require_approval = True
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    calls: list[str] = []
    agent = Agent(
        llm_client=ApprovalClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        mode="plan",
        plan_approval_callback=lambda plan: calls.append(plan.id) or PlanDecision("cancel"),
    )

    async def runner():
        async for _event in agent.run("先执行任务 A，然后执行任务 B"):
            pass

    asyncio.run(runner())

    assert calls, "Agent(mode=plan) must forward the plan approval callback"
