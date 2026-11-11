from __future__ import annotations

import asyncio
from typing import Any

from taskpaw.agent import PlanExecuteAgent
from taskpaw.config import load_config
from taskpaw.plan import ExecutionPlan, Task, TaskType
from taskpaw.plan.verifier import TaskVerifier
from taskpaw.tools import ToolRegistry, get_builtin_tools

PLAN_ONE = '{"tasks":[{"id":"a","description":"任务 A","type":"ANALYSIS"}]}'
PLAN_TWO = (
    '{"tasks":['
    '{"id":"a","description":"任务 A","type":"ANALYSIS","dependencies":[]},'
    '{"id":"b","description":"任务 B","type":"ANALYSIS","dependencies":["a"]}'
    "]}"
)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class VerdictClient:
    """Fake LLM: returns a plan, task output and scripted verification verdicts."""

    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self, plan: str = PLAN_ONE, verdicts: list[bool] | None = None) -> None:
        self.plan = plan
        self.verdicts = verdicts or [True]
        self.verify_calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _text(messages[-1].content)
        if "请为以下目标创建执行计划" in body:
            yield {"type": "text_delta", "text": self.plan}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "子任务验收器" in system_prompt:
            passed = self.verdicts[min(self.verify_calls, len(self.verdicts) - 1)]
            self.verify_calls += 1
            payload = '{"passed": %s, "reason": "验收意见"}' % ("true" if passed else "false")
            yield {"type": "text_delta", "text": payload}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        yield {"type": "usage", "usage": {"input_tokens": 10, "output_tokens": 5}}
        yield {"type": "text_delta", "text": "任务完成"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class JsonClient:
    """Fake LLM returning a fixed verifier JSON payload (unit tests)."""

    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        yield {"type": "text_delta", "text": self.payload}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _task(resources: list[str] | None = None) -> Task:
    return Task("task_1", "写一个文件", TaskType.FILE_WRITE, resources=resources or [])


# ---------------------------------------------------------------------------
# Unit: deterministic + LLM gates
# ---------------------------------------------------------------------------


def test_declared_missing_file_fails_without_calling_the_model(tmp_path):
    client = JsonClient('{"passed": true}')
    verifier = TaskVerifier(client, str(tmp_path))
    plan = ExecutionPlan(id="plan_1", goal="demo")

    verdict = asyncio.run(verifier.verify(plan, _task(["file:missing.py"]), "写好了"))

    assert verdict.passed is False
    assert "missing.py" in verdict.reason
    assert client.calls == 0


def test_declared_existing_file_passes_the_model_gate(tmp_path):
    (tmp_path / "note.txt").write_text("hi", encoding="utf-8")
    client = JsonClient('{"passed": true, "reason": "ok"}')
    verifier = TaskVerifier(client, str(tmp_path))
    plan = ExecutionPlan(id="plan_1", goal="demo")

    verdict = asyncio.run(verifier.verify(plan, _task(["file:note.txt"]), "写好了"))

    assert verdict.passed is True
    assert client.calls == 1


def test_unparseable_verifier_output_fails_open(tmp_path):
    client = JsonClient("这不是 JSON")
    verifier = TaskVerifier(client, str(tmp_path))
    plan = ExecutionPlan(id="plan_1", goal="demo")

    verdict = asyncio.run(verifier.verify(plan, _task(), "任务完成"))

    assert verdict.passed is True


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


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


def _run(agent):
    async def runner():
        text = ""
        events = []
        async for event in agent.run("先执行任务 A，然后执行任务 B"):
            events.append(event)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text, events

    return asyncio.run(runner())


def test_rejected_task_is_retried_and_passive_verdict_ends_the_retry(tmp_path, monkeypatch):
    client = VerdictClient(verdicts=[False, True])
    agent = _agent(tmp_path, monkeypatch, client, max_task_retries=2)

    text, events = _run(agent)

    verdicts = [event for event in events if event.get("type") == "plan_task_verified"]
    assert [event["passed"] for event in verdicts] == [False, True]
    assert client.verify_calls == 2
    assert "已完成 [task_1]" in text


def test_always_rejected_task_fails_and_dependents_are_skipped(tmp_path, monkeypatch):
    client = VerdictClient(plan=PLAN_TWO, verdicts=[False])
    agent = _agent(tmp_path, monkeypatch, client, max_task_retries=1, max_repairs=0)

    text, events = _run(agent)

    assert "失败 [task_1]" in text
    assert "已跳过" in text
    done = next(event for event in events if event.get("type") == "done")
    assert done["stop_reason"] == "plan_failed"


def test_verification_can_be_disabled(tmp_path, monkeypatch):
    client = VerdictClient(verdicts=[False])
    agent = _agent(tmp_path, monkeypatch, client, verify_tasks=False)

    text, _ = _run(agent)

    assert client.verify_calls == 0
    assert "已完成 [task_1]" in text
