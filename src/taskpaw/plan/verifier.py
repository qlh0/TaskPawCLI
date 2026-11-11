"""verifier.py — acceptance checks for Plan-mode subtasks.

A subtask is not "done" just because its ReAct loop ended. Before it is marked
completed it goes through two cheap gates: declared artifacts must exist, and a
single model call judges whether the result actually answers the task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from taskpaw.llm.base import LlmClient
from taskpaw.plan.models import ExecutionPlan, Task
from taskpaw.types import Message, Usage

VERIFIER_PROMPT = """你是 TaskPaw 的子任务验收器。
判断这个子任务是否真正完成了它的目标，只返回以下 JSON：
{
  "passed": true,
  "reason": "一句话结论",
  "issues": ["具体问题，可空"]
}
判断标准：
- 结果必须针对任务描述，而不是泛泛而谈或答非所问；
- 如果任务要求写文件、改代码或执行命令，结果中应体现实际做了什么；
- 声称完成但没有任何证据，或明显漏掉任务要求的步骤 → passed=false；
- 只判断是否完成，不要因为措辞、风格或格式扣分。
"""

_JSON_FENCE = re.compile(r"```(?:json)?\s*")
_JSON_CLOSE = re.compile(r"```")


@dataclass(slots=True)
class Verdict:
    passed: bool
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class TaskVerifier:
    def __init__(self, llm_client: LlmClient, cwd: str) -> None:
        self.llm_client = llm_client
        self.cwd = Path(cwd)

    async def verify(self, plan: ExecutionPlan, task: Task, result_text: str) -> Verdict:
        artifact_failure = self._check_declared_artifacts(task)
        if artifact_failure:
            return Verdict(passed=False, reason=artifact_failure, issues=[artifact_failure])
        return await self._llm_verdict(plan, task, result_text)

    def _check_declared_artifacts(self, task: Task) -> str:
        """Cheap deterministic gate: declared files/dirs must exist."""

        for token in task.resources:
            kind, _, value = str(token).strip().partition(":")
            kind = kind.strip().lower()
            value = value.strip()
            if kind not in {"file", "dir"} or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.cwd / path
            if kind == "dir":
                if not (path.exists() and path.is_dir()):
                    return f"任务声明会产出目录 {value}，但它不存在"
            elif not path.exists():
                return f"任务声明会产出文件 {value}，但它不存在"
        return ""

    async def _llm_verdict(
        self,
        plan: ExecutionPlan,
        task: Task,
        result_text: str,
    ) -> Verdict:
        context = (
            f"目标：{plan.goal}\n"
            f"任务 [{task.id}]：{task.description}\n"
            f"任务类型：{task.type.value}\n"
            f"声明影响的资源：{', '.join(task.resources) or '无'}\n\n"
            f"任务产出：\n{_preview(result_text, 2000)}\n\n"
            "请判断这个子任务是否真正完成。"
        )
        messages = [Message(role="user", content=context)]
        text = ""
        usage = Usage()
        async for event in self.llm_client.chat(messages, [], system_prompt=VERIFIER_PROMPT):
            event_type = event.get("type")
            if event_type == "text_delta":
                text += str(event.get("text") or "")
            elif event_type == "usage":
                usage = usage + Usage.from_mapping(event.get("usage") or {})
            elif event_type == "error":
                raise event["error"]
        return _parse_verdict(text, usage)


def _parse_verdict(text: str, usage: Usage) -> Verdict:
    cleaned = _JSON_CLOSE.sub("", _JSON_FENCE.sub("", text or "")).strip()
    if not cleaned:
        return Verdict(passed=True, reason="验收器没有输出，按通过处理", usage=usage)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return Verdict(passed=True, reason="验收器输出无法解析，按通过处理", usage=usage)
    if not isinstance(data, dict):
        return Verdict(passed=True, reason="验收器输出格式异常，按通过处理", usage=usage)

    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return Verdict(
        passed=bool(data.get("passed", True)),
        reason=str(data.get("reason") or ""),
        issues=[str(item) for item in issues if str(item).strip()],
        usage=usage,
    )


def _preview(text: str, max_len: int) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
