from __future__ import annotations

import asyncio
from typing import Any

from taskpaw.agent import PlanExecuteAgent
from taskpaw.config import load_config
from taskpaw.plan import ExecutionPlan, Task, TaskType
from taskpaw.plan.store import PlanStore
from taskpaw.tools import ToolRegistry, get_builtin_tools


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class OneTaskClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _text(messages[-1].content)
        if "子任务验收器" in system_prompt:
            yield {"type": "text_delta", "text": '{"passed": true, "reason": "ok"}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        if "请为以下目标创建执行计划" in body:
            yield {
                "type": "text_delta",
                "text": '{"tasks":[{"id":"a","description":"任务 A","type":"ANALYSIS"}]}',
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        yield {"type": "text_delta", "text": "完成任务"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _run_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.policy.hitl_mode = "never"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    agent = PlanExecuteAgent(
        llm_client=OneTaskClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def runner():
        async for _event in agent.run("先执行任务 A，然后执行任务 B"):
            pass

    asyncio.run(runner())
    return config


def test_plan_history_is_persisted(tmp_path, monkeypatch):
    _run_plan(tmp_path, monkeypatch)
    store = PlanStore()

    plans = store.list_plans()
    assert len(plans) == 1
    plan = store.get_plan(plans[0]["plan_id"])
    assert plan is not None
    assert plan["status"] == "COMPLETED"
    assert plan["stop_reason"] == "end_turn"
    assert plan["tasks"][0]["status"] == "COMPLETED"
    assert plan["tasks"][0]["result_preview"].startswith("完成任务")

    events = store.list_events(plans[0]["plan_id"])
    types = {event["type"] for event in events}
    assert {"plan_ready", "plan_task_started", "plan_task_done"} <= types


def test_history_retention_keeps_the_newest_plans(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    store = PlanStore(max_plans=2)
    for index in range(3):
        plan = ExecutionPlan(id=f"plan_{index}", goal=f"goal {index}")
        plan.add_task(Task(f"task_{index}", "demo", TaskType.ANALYSIS))
        store.start_plan(plan)

    plans = store.list_plans()
    assert [item["plan_id"] for item in plans] == ["plan_2", "plan_1"]


def test_store_survives_a_missing_history_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    path = tmp_path / "nested" / "deeper" / "plans.db"
    store = PlanStore(path)
    plan = ExecutionPlan(id="plan_x", goal="g")
    plan.add_task(Task("task_1", "demo", TaskType.ANALYSIS))
    store.start_plan(plan)
    assert store.get_plan("plan_x") is not None
