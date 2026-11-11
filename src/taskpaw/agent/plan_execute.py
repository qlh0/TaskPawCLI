from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from taskpaw.agent.query import query
from taskpaw.config import TaskPawConfig
from taskpaw.llm.base import LlmClient
from taskpaw.plan import (
    ExecutionPlan,
    PlanDecision,
    Planner,
    PlanStatus,
    Task,
    TaskStatus,
    resources_conflict,
)
from taskpaw.plan.store import PlanStore
from taskpaw.plan.verifier import TaskVerifier
from taskpaw.prompt import PromptAssembler
from taskpaw.skill import SkillContextBuffer
from taskpaw.snapshot import SnapshotService
from taskpaw.tools.registry import ToolRegistry
from taskpaw.types import Message, Usage


@dataclass(slots=True)
class TaskRunResult:
    task: Task
    text: str
    usage: Usage
    turns: int
    error: Exception | None = None
    stop_reason: str = "end_turn"

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens


class PlanExecuteAgent:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: TaskPawConfig,
        cwd: str,
        approval_callback=None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
        plan_approval_callback: Callable[[ExecutionPlan], PlanDecision | Awaitable[PlanDecision]]
        | None = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.plan_approval_callback = plan_approval_callback
        self.history: list[Message] = []
        self.verifier = TaskVerifier(llm_client, cwd) if config.plan.verify_tasks else None
        self.store = (
            PlanStore(max_plans=config.plan.max_history_plans) if config.plan.keep_history else None
        )

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        total_usage = Usage()
        total_turns = 0
        final_text = ""
        stop_reason = "end_turn"
        plan: ExecutionPlan | None = None
        try:
            yield {"type": "text_delta", "text": f"正在规划任务：{message}\n\n"}
            yield {"type": "plan_status", "phase": "planning"}
            async for event in self.planner.stream_plan(message):
                if event.get("type") == "plan_created":
                    plan = event["plan"]
                    continue
                if event.get("type") == "usage":
                    total_usage = total_usage + Usage.from_mapping(event.get("usage") or {})
                yield event
            if plan is None:
                raise ValueError("planner did not produce an execution plan")
            self._store_start(plan)
            serialized = plan.auto_serialize_conflicts()
            if serialized:
                event = {
                    "type": "plan_serialized",
                    "pairs": [
                        {"earlier": earlier, "later": later} for earlier, later in serialized
                    ],
                }
                self._record(plan.id, event)
                yield event
                yield {
                    "type": "text_delta",
                    "text": "因资源冲突改为串行："
                    + "、".join(f"{later} ← {earlier}" for earlier, later in serialized)
                    + "\n\n",
                }
            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
            ready_event = {
                "type": "plan_ready",
                "plan_id": plan.id,
                "task_count": len(plan.all_tasks()),
                "summary": plan.summarize(),
            }
            self._record(plan.id, ready_event)
            yield ready_event

            cancelled = False
            if self.config.plan.require_approval and self.plan_approval_callback is not None:
                revisions = 0
                while True:
                    decision = await _resolve_plan_decision(self.plan_approval_callback, plan)
                    if decision.action == "cancel":
                        cancelled = True
                        break
                    if decision.action != "revise":
                        break
                    if revisions >= max(0, self.config.plan.max_plan_revisions):
                        yield {
                            "type": "text_delta",
                            "text": "已达到计划修改次数上限，按当前计划执行。\n\n",
                        }
                        break
                    revisions += 1
                    revision_event = {
                        "type": "plan_revision_started",
                        "attempt": revisions,
                        "feedback": decision.feedback,
                    }
                    self._record(plan.id, revision_event)
                    yield revision_event
                    yield {
                        "type": "text_delta",
                        "text": f"按意见重新规划（第 {revisions} 次）……\n\n",
                    }
                    try:
                        plan = await self.planner.create_plan(
                            _revision_goal(message, decision.feedback)
                        )
                    except Exception as exc:  # noqa: BLE001 - keep the previous plan
                        yield {
                            "type": "text_delta",
                            "text": f"重新规划失败：{exc}；继续使用原计划。\n\n",
                        }
                        break
                    total_usage = total_usage + self.planner.last_usage
                    yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
                decision_event = {
                    "type": "plan_cancelled" if cancelled else "plan_approved",
                    "plan_id": plan.id,
                }
                self._record(plan.id, decision_event)
                yield decision_event

            if cancelled:
                plan.mark_cancelled()
                stop_reason = "plan_cancelled"
                final_text += "已取消计划执行，未运行任何任务。\n\n"
                yield {"type": "text_delta", "text": "已取消计划执行，未运行任何任务。\n\n"}
            else:
                async for event in self._execute_plan(plan):
                    if event.get("type") == "usage":
                        total_usage = total_usage + Usage.from_mapping(event.get("usage") or {})
                    elif event.get("type") == "plan_task_done":
                        total_turns += int(event.get("turns") or 0)
                        self._record(plan.id, event)
                        task = plan.tasks.get(str(event.get("task_id") or ""))
                        if task is not None:
                            self._store_task(
                                plan.id,
                                task,
                                tokens=int(event.get("tokens") or 0),
                                turns=int(event.get("turns") or 0),
                            )
                        continue
                    elif event.get("type") == "text_delta":
                        final_text += str(event.get("text") or "")
                    else:
                        self._record(plan.id, event)
                    yield event
                stop_reason = "end_turn" if plan.status == PlanStatus.COMPLETED else "plan_failed"
            self._store_finish(plan, stop_reason=stop_reason, total_usage=total_usage)
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            if plan is not None:
                plan.mark_failed()
                self._store_finish(plan, stop_reason="error", total_usage=total_usage)
            yield {"type": "error", "error": exc}
            return
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")
        done: dict[str, Any] = {
            "type": "done",
            "total_turns": total_turns,
            "total_tokens": total_usage.total_tokens,
            "usage": total_usage.to_dict(),
            "stop_reason": stop_reason,
            "messages": self.history,
        }
        costs = _calculate_costs(self.llm_client, total_usage)
        if costs:
            done["cost"] = costs
        yield done

    def _record(self, plan_id: str, event: dict[str, Any]) -> None:
        if self.store is None:
            return
        # History is best effort: a broken database must never break a run.
        with suppress(Exception):
            self.store.record_event(plan_id, event)

    def _store_start(self, plan: ExecutionPlan) -> None:
        if self.store is None:
            return
        with suppress(Exception):
            self.store.start_plan(plan)

    def _store_task(
        self,
        plan_id: str,
        task: Task,
        *,
        tokens: int = 0,
        turns: int = 0,
    ) -> None:
        if self.store is None:
            return
        with suppress(Exception):
            self.store.upsert_task(plan_id, task, tokens=tokens, turns=turns)

    def _store_finish(
        self,
        plan: ExecutionPlan,
        *,
        stop_reason: str,
        total_usage: Usage,
    ) -> None:
        if self.store is None:
            return
        with suppress(Exception):
            self.store.finish_plan(
                plan,
                stop_reason=stop_reason,
                total_tokens=total_usage.total_tokens,
            )

    async def _execute_plan(self, plan: ExecutionPlan) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text_delta", "text": "开始执行计划……\n\n"}
        yield {"type": "plan_status", "phase": "execution"}
        plan.mark_started()
        repairs = 0
        max_parallel = max(0, int(self.config.plan.max_parallel_tasks))
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        running: dict[str, asyncio.Task] = {}
        results: dict[str, TaskRunResult] = {}
        try:
            while True:
                ready = _executable_tasks_in_order(plan)
                if not ready and not running:
                    if plan.has_failed() and repairs < max(0, self.config.plan.max_repairs):
                        repairs += 1
                        repaired = False
                        async for event in self._repair_plan(plan, attempt=repairs):
                            if event.get("type") == "plan_repair_result":
                                repaired = bool(event.get("ok"))
                            else:
                                yield event
                        if repaired:
                            continue
                    break

                in_flight: set[str] = set()
                for task_id in running:
                    in_flight |= plan.tasks[task_id].effective_resources()
                admitted: list[str] = []
                for task in ready:
                    if max_parallel and len(running) >= max_parallel:
                        break
                    if task.id in running:
                        continue
                    if resources_conflict(task.effective_resources(), in_flight):
                        continue
                    running[task.id] = asyncio.create_task(
                        self._pump_task_events(plan, task, queue)
                    )
                    in_flight |= task.effective_resources()
                    admitted.append(task.id)

                if admitted:
                    yield {
                        "type": "plan_schedule",
                        "limit": max_parallel,
                        "running": len(running),
                        "started": admitted,
                    }
                    if len(admitted) > 1:
                        yield {
                            "type": "text_delta",
                            "text": (
                                f"并行执行（{len(running)}/{max_parallel or '∞'}）："
                                f"{', '.join(admitted)}\n\n"
                            ),
                        }
                    else:
                        yield {
                            "type": "text_delta",
                            "text": f"执行 [{admitted[0]}]……\n\n",
                        }

                if not running:
                    break

                event = await queue.get()
                event_type = event.get("type")
                if event_type == "plan_task_stream_done":
                    task_id = str(event.get("task_id") or "")
                    runner = running.pop(task_id, None)
                    if runner is not None:
                        await asyncio.gather(runner, return_exceptions=True)
                    result = results.pop(task_id, None)
                    if result is None:
                        task = plan.tasks.get(task_id)
                        if task is not None and task.status == TaskStatus.RUNNING:
                            result = TaskRunResult(
                                task=task,
                                text="",
                                usage=Usage(),
                                turns=0,
                                error=RuntimeError(f"task {task_id} produced no result"),
                            )
                    if result is not None:
                        async for applied in self._apply_task_result(result):
                            yield applied
                elif event_type == "plan_task_result":
                    result = event["result"]
                    results[result.task.id] = result
                else:
                    yield event
        finally:
            for runner in running.values():
                if not runner.done():
                    runner.cancel()
            await asyncio.gather(*running.values(), return_exceptions=True)

        skipped = plan.skip_blocked()
        if skipped:
            yield {
                "type": "text_delta",
                "text": f"因依赖失败被跳过：{', '.join(task.id for task in skipped)}\n\n",
            }

        if plan.is_all_completed():
            plan.mark_completed()
            title = "计划执行完成。"
        elif plan.has_failed():
            plan.mark_failed()
            title = "计划部分完成，但有任务执行失败。"
        else:
            plan.mark_failed()
            title = "计划已停滞，因为任务依赖未满足。"
        yield {"type": "text_delta", "text": f"{title}\n\n{_build_plan_result(plan)}"}

    async def _apply_task_result(self, result: TaskRunResult) -> AsyncIterator[dict[str, Any]]:
        if result.error:
            result.task.mark_failed(str(result.error))
            yield {"type": "text_delta", "text": f"失败 [{result.task.id}]：{result.error}\n\n"}
        else:
            result.task.mark_completed(result.text)
            yield {
                "type": "text_delta",
                "text": f"已完成 [{result.task.id}]：{_preview(result.text)}\n\n",
            }
        yield {"type": "usage", "usage": result.usage.to_dict()}
        yield {
            "type": "plan_task_done",
            "task_id": result.task.id,
            "turns": result.turns,
            "tokens": result.tokens,
            "failed": bool(result.error),
        }

    async def _repair_plan(
        self,
        plan: ExecutionPlan,
        *,
        attempt: int,
    ) -> AsyncIterator[dict[str, Any]]:
        failed = [task for task in plan.all_tasks() if task.status == TaskStatus.FAILED]
        failed_ids = [task.id for task in failed]
        reason = "; ".join(f"{task.id}: {task.error}" for task in failed) or "unknown failure"
        yield {
            "type": "text_delta",
            "text": f"局部重规划（第 {attempt} 次）：替换失败任务 {', '.join(failed_ids)}……\n\n",
        }
        yield {"type": "plan_repair_started", "attempt": attempt, "failed_ids": failed_ids}
        try:
            replacement, usage = await self.planner.repair(
                plan,
                failed_ids,
                reason,
                attempt=attempt,
            )
        except Exception as exc:  # noqa: BLE001 - a failed repair must not abort the run
            yield {"type": "plan_repair_result", "ok": False, "error": str(exc)}
            yield {"type": "text_delta", "text": f"局部重规划失败：{exc}\n\n"}
            return

        yield {"type": "usage", "usage": usage.to_dict()}
        if not replacement:
            yield {"type": "plan_repair_result", "ok": False, "error": "empty replacement"}
            yield {
                "type": "text_delta",
                "text": "局部重规划没有返回可用任务，保留原计划。\n\n",
            }
            return

        added = plan.apply_repair(replacement, attempt=attempt)
        if not added:
            yield {"type": "plan_repair_result", "ok": False, "error": "merge failed"}
            yield {
                "type": "text_delta",
                "text": "局部重规划结果无法合并（存在环或依赖非法），保留原计划。\n\n",
            }
            return

        yield {
            "type": "plan_repair_result",
            "ok": True,
            "task_ids": [task.id for task in added],
        }
        yield {
            "type": "plan_repaired",
            "attempt": attempt,
            "task_ids": [task.id for task in added],
        }
        yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

    async def _pump_task_events(
        self,
        plan: ExecutionPlan,
        task: Task,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        try:
            async for event in self._execute_task_stream(plan, task):
                await queue.put(event)
        finally:
            await queue.put({"type": "plan_task_stream_done", "task_id": task.id})

    async def _execute_task_stream(
        self,
        plan: ExecutionPlan,
        task: Task,
    ) -> AsyncIterator[dict[str, Any]]:
        task.mark_started()
        yield {
            "type": "plan_task_started",
            "task_id": task.id,
            "task_description": task.description,
        }

        max_retries = max(0, int(self.config.plan.max_task_retries))
        total_usage = Usage()
        total_turns = 0
        attempts = 0
        retry_context = ""
        last_error: Exception | None = None
        last_stop_reason = "end_turn"

        while attempts <= max_retries:
            attempts += 1
            task.attempts = attempts
            if attempts > 1:
                reason_text = str(last_error or f"stop_reason={last_stop_reason}")
                yield {
                    "type": "plan_task_retry",
                    "task_id": task.id,
                    "attempt": attempts,
                    "max_attempts": max_retries + 1,
                    "reason": reason_text,
                }
                yield {
                    "type": "text_delta",
                    "text": (
                        f"重试 [{task.id}] 第 {attempts}/{max_retries + 1} 次："
                        f"{_preview(reason_text, 120)}\n\n"
                    ),
                }

            attempt_result: TaskRunResult | None = None
            async for event in self._run_task_attempt(plan, task, retry_context):
                if event.get("type") == "plan_task_result":
                    attempt_result = event["result"]
                else:
                    yield event
            if attempt_result is None:
                attempt_result = TaskRunResult(
                    task=task,
                    text="",
                    usage=Usage(),
                    turns=0,
                    error=RuntimeError(f"task {task.id} produced no result"),
                )

            total_usage = total_usage + attempt_result.usage
            total_turns += attempt_result.turns
            last_error = attempt_result.error
            last_stop_reason = attempt_result.stop_reason

            finished = (
                attempt_result.error is None
                and attempt_result.stop_reason == "end_turn"
                and bool(attempt_result.text.strip())
            )

            if finished and self.verifier is not None:
                verdict = await self.verifier.verify(plan, task, attempt_result.text)
                if verdict.usage.total_tokens:
                    total_usage = total_usage + verdict.usage
                yield {
                    "type": "plan_task_verified",
                    "task_id": task.id,
                    "passed": verdict.passed,
                    "reason": verdict.reason,
                }
                if not verdict.passed:
                    reason = verdict.reason or "未通过验收"
                    if verdict.issues:
                        reason = f"{reason}（{'；'.join(verdict.issues[:3])}）"
                    attempt_result = TaskRunResult(
                        task=task,
                        text=attempt_result.text,
                        usage=attempt_result.usage,
                        turns=attempt_result.turns,
                        error=RuntimeError(f"验收未通过：{reason}"),
                        stop_reason="verify_rejected",
                    )
                    last_error = attempt_result.error
                    last_stop_reason = attempt_result.stop_reason
                    finished = False

            if finished:
                yield {
                    "type": "plan_task_result",
                    "result": TaskRunResult(
                        task=task,
                        text=attempt_result.text,
                        usage=total_usage,
                        turns=total_turns,
                    ),
                }
                return

            if not self._should_retry(attempt_result):
                break
            retry_context = _retry_context(attempt_result)

        error = last_error or RuntimeError(f"task stopped early: {last_stop_reason}")
        yield {
            "type": "plan_task_result",
            "result": TaskRunResult(
                task=task,
                text="",
                usage=total_usage,
                turns=total_turns,
                error=error,
                stop_reason=last_stop_reason,
            ),
        }

    async def _run_task_attempt(
        self,
        plan: ExecutionPlan,
        task: Task,
        retry_context: str,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "plan_task_attempt",
            "task_id": task.id,
            "attempt": task.attempts,
        }
        text = ""
        tool_results: list[str] = []
        usage = Usage()
        stream_usage = Usage()
        turns = 0
        stop_reason = "end_turn"
        user_message = _task_context(plan, task)
        if retry_context:
            user_message = f"{user_message}\n\n{retry_context}"
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._task_system_prompt(plan, task),
                user_message=user_message,
                history=[],
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                # Parallel plan tasks must never share one-shot Skill context.
                skill_context_buffer=SkillContextBuffer(),
                max_turns=self.max_task_turns,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    text += str(event.get("text") or "")
                elif event_type == "tool_result":
                    content = str(event.get("result") or "")
                    if content:
                        tool_results.append(content)
                    yield _with_task_context(event, task)
                elif event_type in {
                    "thinking_delta",
                    "tool_call",
                    "context_compressed",
                    "loop_warning",
                    "loop_extended",
                    "loop_wrapup",
                }:
                    yield _with_task_context(event, task)
                elif event_type == "usage":
                    stream_usage = stream_usage + Usage.from_mapping(event.get("usage") or {})
                elif event_type == "done":
                    turns += int(event.get("total_turns") or 0)
                    done_usage = Usage.from_mapping(event.get("usage") or {})
                    usage = done_usage if done_usage.total_tokens else stream_usage
                    stop_reason = str(event.get("stop_reason") or "end_turn")
                elif event_type == "error":
                    raise event["error"]
            result_text = text.strip() or "\n".join(tool_results).strip()
            yield {
                "type": "plan_task_result",
                "result": TaskRunResult(
                    task=task,
                    text=result_text,
                    usage=usage,
                    turns=turns,
                    stop_reason=stop_reason,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "plan_task_result",
                "result": TaskRunResult(
                    task=task,
                    text="",
                    usage=usage if usage.total_tokens else stream_usage,
                    turns=turns,
                    error=exc,
                    stop_reason=stop_reason,
                ),
            }

    @staticmethod
    def _should_retry(result: TaskRunResult) -> bool:
        """Fatal outcomes (budget, policy) are not worth retrying."""

        if result.stop_reason in {"budget_exhausted", "hard_cap"}:
            return False
        text = str(result.error or "").lower()
        fatal_markers = (
            "permission denied",
            "access denied",
            "拒绝访问",
            "blocked by policy",
            "command guard",
            "path guard",
        )
        return not any(marker in text for marker in fatal_markers)

    def _task_system_prompt(self, plan: ExecutionPlan, task: Task) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_static()
        return (
            base
            + "\n\n你正在执行 Plan-and-Execute DAG 中的一个任务。\n"
            + f"任务 id：{task.id}\n任务类型：{task.type.value}\n"
            + "请具体完成任务，并在需要时使用工具。"
            + _task_language_instruction(plan.goal)
        )


def _with_task_context(event: dict[str, Any], task: Task) -> dict[str, Any]:
    return {
        **event,
        "phase": "execution",
        "task_id": task.id,
        "task_description": task.description,
    }


def _executable_tasks_in_order(plan: ExecutionPlan) -> list[Task]:
    executable_ids = {task.id for task in plan.executable_tasks()}
    return [plan.tasks[task_id] for task_id in plan.execution_order() if task_id in executable_ids]


def _task_context(plan: ExecutionPlan, task: Task) -> str:
    lines = [
        f"目标：{plan.goal}",
        f"当前任务 [{task.id}]：{task.description}",
        "",
        "已完成的依赖任务结果：",
    ]
    for dep_id in task.dependencies:
        dep = plan.get_task(dep_id)
        if dep and dep.status == TaskStatus.COMPLETED:
            lines.append(f"- [{dep.id}] {dep.description}: {_preview(dep.result, 800)}")
    return "\n".join(lines)


async def _resolve_plan_decision(callback, plan: ExecutionPlan) -> PlanDecision:
    result = callback(plan)
    if isawaitable(result):
        result = await result
    if isinstance(result, PlanDecision):
        return result
    if isinstance(result, str) and result in {"approve", "cancel", "revise"}:
        return PlanDecision(result)
    return PlanDecision("approve")


def _revision_goal(message: str, feedback: str) -> str:
    note = feedback.strip() or "请调整任务拆分、顺序或范围。"
    return f"{message}\n\n用户看过第一版计划后提出了修改意见：{note}\n请据此重新给出完整计划。"


def _retry_context(result: TaskRunResult) -> str:
    reason = (
        str(result.error)
        if result.error
        else f"上一次尝试未正常结束（stop_reason={result.stop_reason}）"
    )
    if result.stop_reason == "verify_rejected":
        return (
            "上一轮产出未通过验收，原因：\n"
            f"{_preview(reason, 600)}\n\n"
            "请针对验收意见补齐缺失的步骤或证据，重新完成这个任务；"
            "如果已经写过文件，先检查它们的当前内容再修改。"
        )
    return (
        "上一次尝试失败，原因：\n"
        f"{_preview(reason, 600)}\n\n"
        "请先检查工作区当前状态（可能已经部分写入或已有副作用），换一种做法再试；"
        "不要重复上一次失败的操作。"
    )


def _build_plan_result(plan: ExecutionPlan) -> str:
    status_labels = {
        TaskStatus.PENDING: "待执行",
        TaskStatus.RUNNING: "执行中",
        TaskStatus.COMPLETED: "已完成",
        TaskStatus.FAILED: "失败",
        TaskStatus.SKIPPED: "已跳过",
    }
    counts = plan.status_counts()
    lines = [
        f"任务摘要：共 {len(plan.all_tasks())} 个任务，"
        f"已完成 {counts['COMPLETED']}，失败 {counts['FAILED']}，跳过 {counts['SKIPPED']}。",
        "",
    ]
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {status_labels[task.status]}：{task.description}")
        if task.status == TaskStatus.FAILED and task.error:
            lines.append(f"  失败原因：{_preview(task.error, 200)}")
        elif task.result:
            lines.append(f"  结果：{_preview(task.result)}")
    return "\n".join(lines) + "\n"


def _task_language_instruction(goal: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in goal):
        return "\n用户目标包含中文；所有进度说明、分析和最终结果都必须使用中文。"
    return "\nUse the same language as the user's goal for all progress and results."


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _calculate_costs(llm_client: LlmClient, usage: Usage) -> dict[str, Any]:
    calculator = getattr(llm_client, "calculate_cost", None)
    if not callable(calculator):
        return {}
    result: dict[str, Any] = {}
    for currency in ("usd", "cny"):
        try:
            breakdown = calculator(usage, currency=currency)
        except (KeyError, TypeError, ValueError):
            continue
        result[currency] = breakdown.to_dict()
    return result
