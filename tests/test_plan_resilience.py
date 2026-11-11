from __future__ import annotations

import asyncio
import re
from typing import Any

from taskpaw.agent import PlanExecuteAgent
from taskpaw.config import load_config
from taskpaw.plan import ExecutionPlan, Task, TaskStatus, TaskType
from taskpaw.tools import ToolRegistry, get_builtin_tools

PLAN_ONE_TASK = (
    '{"summary":"one","tasks":[{"id":"a","description":"任务 A","type":"ANALYSIS",'
    '"dependencies":[]}]}'
)
PLAN_TWO_TASKS = (
    '{"summary":"two","tasks":['
    '{"id":"a","description":"任务 A","type":"ANALYSIS","dependencies":[]},'
    '{"id":"b","description":"任务 B","type":"ANALYSIS","dependencies":["a"]}'
    "]}"
)
REPAIR_ONE_TASK = (
    '{"tasks":[{"id":"fix","description":"修复任务","type":"ANALYSIS","dependencies":[]}]}'
)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _task_id_from_prompt(system_prompt: str) -> str:
    match = re.search(r"任务 id：(\S+)", system_prompt)
    return match.group(1) if match else "unknown"


class ScriptedClient:
    """Fake LLM: returns a fixed plan, a fixed repair and scripted task outcomes."""

    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(
        self,
        *,
        plan: str,
        repair: str | None = None,
        outcomes: dict[str, list[str]] | None = None,
    ) -> None:
        self.plan = plan
        self.repair = repair
        self.outcomes = outcomes or {}
        self.calls: dict[str, int] = {}
        self.user_messages: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _text(messages[-1].content)
        self.user_messages.append(body)

        if "请为以下目标创建执行计划" in body:
            yield {"type": "text_delta", "text": self.plan}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "局部修复规划器" in system_prompt:
            yield {"type": "text_delta", "text": self.repair or '{"tasks": []}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        task_id = _task_id_from_prompt(system_prompt)
        index = self.calls.get(task_id, 0)
        self.calls[task_id] = index + 1
        script = self.outcomes.get(task_id) or ["ok"]
        action = script[min(index, len(script) - 1)]

        if action == "guard":
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": f"call_{index}",
                    "function": {"name": "read_file", "arguments": '{"path":"missing.txt"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return

        yield {"type": "usage", "usage": {"input_tokens": 10, "output_tokens": 5}}
        if action == "raise":
            raise RuntimeError("simulated provider failure")

        yield {"type": "text_delta", "text": f"{task_id} 完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _agent(tmp_path, monkeypatch, client, **loop_overrides):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    for key, value in loop_overrides.items():
        setattr(config.loop, key, value)
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


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def test_subtask_retries_until_it_succeeds(tmp_path, monkeypatch):
    client = ScriptedClient(plan=PLAN_ONE_TASK, outcomes={"task_1": ["raise", "raise", "ok"]})
    agent = _agent(tmp_path, monkeypatch, client)
    agent.config.plan.max_task_retries = 3

    text, events = _run(agent)

    assert client.calls["task_1"] == 3
    assert "已完成 [task_1]" in text
    retries = [event for event in events if event.get("type") == "plan_task_retry"]
    assert [event["attempt"] for event in retries] == [2, 3]
    assert _done(events)["stop_reason"] == "end_turn"
    assert any("上一次尝试失败" in message for message in client.user_messages)


def test_failed_attempts_still_count_tokens(tmp_path, monkeypatch):
    client = ScriptedClient(plan=PLAN_ONE_TASK, outcomes={"task_1": ["raise", "ok"]})
    agent = _agent(tmp_path, monkeypatch, client)
    agent.config.plan.max_task_retries = 1

    _, events = _run(agent)

    # The failed attempt emits 15 tokens, the successful one another 15.
    assert _done(events)["total_tokens"] == 30


def test_guard_terminated_subtask_is_treated_as_failure(tmp_path, monkeypatch):
    client = ScriptedClient(plan=PLAN_ONE_TASK, outcomes={"task_1": ["guard", "ok"]})
    agent = _agent(tmp_path, monkeypatch, client, repeat_warn=1, repeat_abort=1)
    agent.config.plan.max_task_retries = 0

    text, events = _run(agent)

    assert "失败 [task_1]" in text
    assert "已完成 [task_1]" not in text
    assert _done(events)["stop_reason"] == "plan_failed"


# ---------------------------------------------------------------------------
# Local repair
# ---------------------------------------------------------------------------


def test_exhausted_retries_trigger_local_repair(tmp_path, monkeypatch):
    client = ScriptedClient(
        plan=PLAN_ONE_TASK,
        repair=REPAIR_ONE_TASK,
        outcomes={"task_1": ["raise"], "fix": ["ok"]},
    )
    agent = _agent(tmp_path, monkeypatch, client)
    agent.config.plan.max_task_retries = 1
    agent.config.plan.max_repairs = 1

    text, events = _run(agent)

    repaired = [event for event in events if event.get("type") == "plan_repaired"]
    assert repaired and repaired[0]["task_ids"] == ["fix"]
    assert "已完成 [fix]" in text
    assert "计划执行完成" in text
    assert _done(events)["stop_reason"] == "end_turn"


def test_repair_can_be_disabled_and_dependents_are_skipped(tmp_path, monkeypatch):
    client = ScriptedClient(plan=PLAN_TWO_TASKS, outcomes={"task_1": ["raise"]})
    agent = _agent(tmp_path, monkeypatch, client)
    agent.config.plan.max_task_retries = 0
    agent.config.plan.max_repairs = 0

    text, events = _run(agent)

    assert "失败 [task_1]" in text
    assert "已跳过" in text
    assert "已完成 0，失败 1，跳过 1" in text
    assert _done(events)["stop_reason"] == "plan_failed"


def test_repair_merge_rolls_back_on_cycle():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    plan.add_task(Task("task_1", "read", TaskType.FILE_READ))
    plan.add_task(Task("task_2", "write", TaskType.FILE_WRITE, ["task_1"]))
    plan.get_task("task_1").mark_completed("done")
    plan.get_task("task_2").mark_failed("boom")

    cyclic = [Task("x", "self cycle", TaskType.ANALYSIS, ["x"])]
    added = plan.apply_repair(cyclic, attempt=1)

    assert added == []
    assert set(plan.tasks) == {"task_1", "task_2"}
    assert plan.get_task("task_2").status == TaskStatus.FAILED
