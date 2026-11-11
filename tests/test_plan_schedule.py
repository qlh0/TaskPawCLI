from __future__ import annotations

import asyncio
import re
from typing import Any

from taskpaw.agent import PlanExecuteAgent
from taskpaw.config import load_config
from taskpaw.plan import ExecutionPlan, Task, TaskType, resources_conflict
from taskpaw.tools import ToolRegistry, get_builtin_tools


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _plan_json(tasks: list[dict]) -> str:
    import json

    return json.dumps({"summary": "schedule", "tasks": tasks})


class ScheduleClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self, plan_json: str, *, delay: float = 0.05) -> None:
        self.plan_json = plan_json
        self.delay = delay
        self.running: set[str] = set()
        self.peak = 0
        self.order: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            yield {"type": "text_delta", "text": self.plan_json}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        match = re.search(r"当前任务 \[(task_\d+)\]", body)
        name = match.group(1) if match else "unknown"
        self.running.add(name)
        self.order.append(name)
        self.peak = max(self.peak, len(self.running))
        await asyncio.sleep(self.delay)
        self.running.discard(name)
        yield {"type": "text_delta", "text": f"{name} 完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _agent(tmp_path, monkeypatch, client, **plan_overrides):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    config.policy.hitl_mode = "never"
    for key, value in plan_overrides.items():
        setattr(config.plan, key, value)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return PlanExecuteAgent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )


def _run(agent, goal: str = "先执行任务 A，然后执行任务 B"):
    async def runner():
        events = []
        async for event in agent.run(goal):
            events.append(event)
            if event.get("type") == "error":
                raise event["error"]
        return events

    return asyncio.run(runner())


# ---------------------------------------------------------------------------
# Resource model
# ---------------------------------------------------------------------------


def test_resources_conflict_understands_files_dirs_and_workspace():
    assert resources_conflict({"file:src/a.py"}, {"file:src/a.py"})
    assert not resources_conflict({"file:src/a.py"}, {"file:src/b.py"})
    assert resources_conflict({"file:src/a.py"}, {"dir:src/"})
    assert resources_conflict({"dir:src/"}, {"dir:src/tools/"})
    assert resources_conflict({"workspace"}, {"file:src/a.py"})
    assert not resources_conflict(set(), {"file:src/a.py"})


def test_auto_serialize_adds_dependencies_for_shared_files():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    plan.add_task(Task("task_1", "write a", TaskType.FILE_WRITE, resources=["file:src/a.py"]))
    plan.add_task(Task("task_2", "write a again", TaskType.FILE_WRITE, resources=["file:src/a.py"]))
    plan.compute_execution_order()

    added = plan.auto_serialize_conflicts()

    assert added == [("task_1", "task_2")]
    assert plan.get_task("task_2").dependencies == ["task_1"]
    assert len(plan.execution_batches()) == 2


def test_auto_serialize_skips_independent_files():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    plan.add_task(Task("task_1", "write a", TaskType.FILE_WRITE, resources=["file:a.py"]))
    plan.add_task(Task("task_2", "write b", TaskType.FILE_WRITE, resources=["file:b.py"]))
    plan.compute_execution_order()

    assert plan.auto_serialize_conflicts() == []


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_parallel_limit_caps_concurrency(tmp_path, monkeypatch):
    client = ScheduleClient(
        _plan_json(
            [
                {"id": "a", "description": "任务 A", "type": "ANALYSIS"},
                {"id": "b", "description": "任务 B", "type": "ANALYSIS"},
                {"id": "c", "description": "任务 C", "type": "ANALYSIS"},
            ]
        )
    )
    agent = _agent(tmp_path, monkeypatch, client, max_parallel_tasks=2)

    events = _run(agent)

    assert client.peak == 2
    assert sorted(client.order) == ["task_1", "task_2", "task_3"]
    assert any(event.get("type") == "plan_schedule" for event in events)


def test_conflicting_tasks_never_run_together(tmp_path, monkeypatch):
    client = ScheduleClient(
        _plan_json(
            [
                {
                    "id": "a",
                    "description": "任务 A",
                    "type": "FILE_WRITE",
                    "resources": ["file:src/shared.py"],
                },
                {
                    "id": "b",
                    "description": "任务 B",
                    "type": "FILE_WRITE",
                    "resources": ["file:src/shared.py"],
                },
            ]
        )
    )
    agent = _agent(tmp_path, monkeypatch, client, max_parallel_tasks=4)

    events = _run(agent)

    assert client.peak == 1
    assert any(event.get("type") == "plan_serialized" for event in events)


def test_serial_mode_runs_one_task_at_a_time(tmp_path, monkeypatch):
    client = ScheduleClient(
        _plan_json(
            [
                {"id": "a", "description": "任务 A", "type": "ANALYSIS"},
                {"id": "b", "description": "任务 B", "type": "ANALYSIS"},
            ]
        )
    )
    agent = _agent(tmp_path, monkeypatch, client, max_parallel_tasks=1)

    _run(agent)

    assert client.peak == 1


def test_zero_parallel_limit_means_unlimited(tmp_path, monkeypatch):
    client = ScheduleClient(
        _plan_json(
            [
                {"id": "a", "description": "任务 A", "type": "ANALYSIS"},
                {"id": "b", "description": "任务 B", "type": "ANALYSIS"},
                {"id": "c", "description": "任务 C", "type": "ANALYSIS"},
            ]
        )
    )
    agent = _agent(tmp_path, monkeypatch, client, max_parallel_tasks=0)

    _run(agent)

    assert client.peak == 3
