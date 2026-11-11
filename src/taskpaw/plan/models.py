from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class TaskType(StrEnum):
    PLANNING = "PLANNING"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    COMMAND = "COMMAND"
    ANALYSIS = "ANALYSIS"
    VERIFICATION = "VERIFICATION"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class PlanDecision:
    """What the human decided about a freshly created plan."""

    action: Literal["approve", "cancel", "revise"] = "approve"
    feedback: str = ""


@dataclass(slots=True)
class Task:
    id: str
    description: str
    type: TaskType = TaskType.ANALYSIS
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attempts: int = 0

    def add_dependency(self, task_id: str) -> None:
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def add_dependent(self, task_id: str) -> None:
        if task_id not in self.dependents:
            self.dependents.append(task_id)

    def mark_started(self) -> None:
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.attempts += 1

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.end_time = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.end_time = time.time()

    def mark_pending(self) -> None:
        """Put a failed task back into the queue for another attempt."""

        self.status = TaskStatus.PENDING
        self.error = ""
        self.end_time = 0.0

    def mark_skipped(self) -> None:
        self.status = TaskStatus.SKIPPED
        self.end_time = time.time()

    def is_executable(self, all_tasks: dict[str, Task]) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
        return all(
            dep_id in all_tasks and all_tasks[dep_id].status == TaskStatus.COMPLETED
            for dep_id in self.dependencies
        )


@dataclass(slots=True)
class ExecutionPlan:
    id: str
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.CREATED
    summary: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    _execution_order: list[str] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task
        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep:
                dep.add_dependent(task.id)
        for existing in self.tasks.values():
            if task.id in existing.dependencies:
                task.add_dependent(existing.id)
        self._execution_order.clear()

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def all_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    def root_tasks(self) -> list[Task]:
        return [task for task in self.tasks.values() if not task.dependencies]

    def executable_tasks(self) -> list[Task]:
        return [task for task in self.tasks.values() if task.is_executable(self.tasks)]

    def compute_execution_order(self) -> bool:
        order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(task: Task) -> bool:
            if task.id in visiting:
                return False
            if task.id in visited:
                return True
            visiting.add(task.id)
            for dep_id in task.dependencies:
                dep = self.tasks.get(dep_id)
                if dep and not visit(dep):
                    return False
            visiting.remove(task.id)
            visited.add(task.id)
            order.append(task.id)
            return True

        for task in self.tasks.values():
            if task.id not in visited and not visit(task):
                return False
        self._execution_order = order
        return True

    def execution_order(self) -> list[str]:
        if not self._execution_order:
            self.compute_execution_order()
        return list(self._execution_order)

    def execution_batches(self) -> list[list[Task]]:
        remaining = set(self.tasks)
        completed: set[str] = set()
        batches: list[list[Task]] = []
        while remaining:
            batch_ids = [
                task_id
                for task_id in self.execution_order()
                if task_id in remaining
                and all(dep_id in completed for dep_id in self.tasks[task_id].dependencies)
            ]
            if not batch_ids:
                break
            batch = [self.tasks[task_id] for task_id in batch_ids]
            batches.append(batch)
            completed.update(batch_ids)
            remaining.difference_update(batch_ids)
        return batches

    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        completed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def is_all_completed(self) -> bool:
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks.values())

    def has_failed(self) -> bool:
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            counts[task.status.value] += 1
        return counts

    def descendants(self, task_ids: Iterable[str]) -> set[str]:
        """All tasks reachable from *task_ids* through dependent edges."""

        found: set[str] = set()
        stack = [task_id for task_id in task_ids if task_id in self.tasks]
        while stack:
            current = stack.pop()
            for dependent_id in self.tasks[current].dependents:
                if dependent_id in self.tasks and dependent_id not in found:
                    found.add(dependent_id)
                    stack.append(dependent_id)
        return found

    def remove_tasks(self, task_ids: Iterable[str]) -> None:
        doomed = {task_id for task_id in task_ids if task_id in self.tasks}
        if not doomed:
            return
        for task_id in doomed:
            self.tasks.pop(task_id, None)
        for task in self.tasks.values():
            task.dependencies = [dep for dep in task.dependencies if dep not in doomed]
            task.dependents = [dep for dep in task.dependents if dep not in doomed]
        self._execution_order.clear()

    def skip_blocked(self) -> list[Task]:
        """Mark pending tasks whose dependencies failed or were skipped."""

        blocked = {TaskStatus.FAILED, TaskStatus.SKIPPED}
        skipped: list[Task] = []
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                if any(
                    dependency not in self.tasks or self.tasks[dependency].status in blocked
                    for dependency in task.dependencies
                ):
                    task.mark_skipped()
                    skipped.append(task)
                    changed = True
        return skipped

    def apply_repair(self, replacement: list[Task], *, attempt: int = 1) -> list[Task]:
        """Replace the failed subgraph with *replacement* tasks.

        Returns the added tasks, or an empty list when the repair cannot be
        merged (for example when it would introduce a cycle).
        """

        failed = {task.id for task in self.tasks.values() if task.status == TaskStatus.FAILED}
        if not failed or not replacement:
            return []

        removed = failed | self.descendants(failed)
        backup = copy.deepcopy(self.tasks)
        used_ids = set(self.tasks) - removed
        id_map: dict[str, str] = {}
        for task in replacement:
            new_id = task.id
            if new_id in used_ids:
                new_id = f"r{attempt}_{task.id}"
            while new_id in used_ids:
                new_id = f"r{attempt}_{new_id}"
            used_ids.add(new_id)
            id_map[task.id] = new_id

        self.remove_tasks(removed)
        added: list[Task] = []
        for task in replacement:
            dependencies = [
                id_map.get(dependency, dependency)
                for dependency in task.dependencies
                if id_map.get(dependency, dependency) in self.tasks
                or id_map.get(dependency, dependency) in id_map.values()
            ]
            new_task = Task(
                id=id_map[task.id],
                description=task.description,
                type=task.type,
                dependencies=dependencies,
            )
            self.add_task(new_task)
            added.append(new_task)

        if not self.compute_execution_order():
            self.tasks = backup
            self._execution_order.clear()
            return []
        return added

    def mark_started(self) -> None:
        self.status = PlanStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self) -> None:
        self.status = PlanStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self) -> None:
        self.status = PlanStatus.FAILED
        self.end_time = time.time()

    def mark_cancelled(self) -> None:
        self.status = PlanStatus.CANCELLED
        self.end_time = time.time()

    def summarize(self) -> str:
        batches = self.execution_batches()
        first_batch = ", ".join(task.id for task in batches[0]) if batches else "无"
        final_batch = ", ".join(task.id for task in batches[-1]) if batches else "无"
        return (
            f"计划 {self.id}：{self.summary or self.goal}\n"
            f"任务数：{len(self.tasks)} | 并行批次：{len(batches)} | "
            f"当前可执行：{len(self.executable_tasks())}\n"
            f"首批任务：{first_batch}\n"
            f"最终汇总：{final_batch}"
        )
