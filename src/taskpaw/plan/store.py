"""store.py 閳?durable Plan-mode history (plans, tasks, events).

The store is intentionally small: it records what happened, not how to resume.
Execution state that would be needed for a safe resume (task-level write sets,
workspace fingerprints) is out of scope for now.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskpaw.plan.models import ExecutionPlan, Task

# Only structural events are persisted; text/thinking deltas would flood the table.
LOGGABLE_EVENTS = frozenset(
    {
        "plan_status",
        "plan_ready",
        "plan_approved",
        "plan_cancelled",
        "plan_serialized",
        "plan_revision_started",
        "plan_schedule",
        "plan_task_started",
        "plan_task_attempt",
        "plan_task_retry",
        "plan_task_done",
        "plan_repair_started",
        "plan_repair_result",
        "plan_repaired",
        "tool_call",
        "tool_result",
        "loop_warning",
        "loop_wrapup",
        "context_compressed",
        "error",
    }
)

_SCHEMA = """
create table if not exists plans (
    plan_id text primary key,
    goal text not null,
    status text not null,
    created_at text not null,
    finished_at text not null default '',
    revisions integer not null default 0,
    repairs integer not null default 0,
    total_tokens integer not null default 0,
    stop_reason text not null default ''
);
create table if not exists plan_tasks (
    plan_id text not null,
    task_id text not null,
    description text not null default '',
    type text not null default '',
    resources text not null default '',
    dependencies text not null default '',
    status text not null default '',
    attempts integer not null default 0,
    error text not null default '',
    result_preview text not null default '',
    tokens integer not null default 0,
    turns integer not null default 0,
    started_at text not null default '',
    ended_at text not null default '',
    primary key (plan_id, task_id)
);
create table if not exists plan_events (
    id integer primary key autoincrement,
    plan_id text not null,
    task_id text not null default '',
    type text not null,
    payload text not null default '{}',
    created_at text not null
);
create index if not exists idx_plan_events_plan on plan_events(plan_id, id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PlanStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_plans: int = 50,
    ) -> None:
        self.path = Path(path or Path.home() / ".taskpaw" / "plans.db").expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_plans = max(1, max_plans)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def start_plan(self, plan: ExecutionPlan) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert or replace into plans
                    (plan_id, goal, status, created_at, finished_at, total_tokens, stop_reason)
                values (?, ?, ?, ?, '', 0, '')
                """,
                (plan.id, plan.goal, plan.status.value, _now()),
            )
            for task in plan.all_tasks():
                self._upsert_task(conn, plan.id, task)
            conn.execute(
                """
                delete from plans where plan_id not in (
                    select plan_id from plans order by rowid desc limit ?
                )
                """,
                (self.max_plans,),
            )
            conn.execute("delete from plan_events where plan_id not in (select plan_id from plans)")
            conn.execute("delete from plan_tasks where plan_id not in (select plan_id from plans)")

    def finish_plan(self, plan: ExecutionPlan, *, stop_reason: str, total_tokens: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                update plans set status = ?, finished_at = ?, stop_reason = ?, total_tokens = ?
                where plan_id = ?
                """,
                (plan.status.value, _now(), stop_reason, total_tokens, plan.id),
            )
            for task in plan.all_tasks():
                self._upsert_task(conn, plan.id, task)

    def upsert_task(self, plan_id: str, task: Task, *, tokens: int = 0, turns: int = 0) -> None:
        with self._connect() as conn:
            self._upsert_task(conn, plan_id, task, tokens=tokens, turns=turns)

    def record_event(self, plan_id: str, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type not in LOGGABLE_EVENTS:
            return
        task_id = str(event.get("task_id") or "")
        payload = _compact({key: value for key, value in event.items() if key != "type"})
        with self._connect() as conn:
            conn.execute(
                "insert into plan_events(plan_id, task_id, type, payload, created_at) "
                "values (?, ?, ?, ?, ?)",
                (plan_id, task_id, event_type, json.dumps(payload, ensure_ascii=False), _now()),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_plans(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select p.*,
                       (select count(*) from plan_tasks t where t.plan_id = p.plan_id)
                           as task_count
                from plans p
                order by p.rowid desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("select * from plans where plan_id = ?", (plan_id,)).fetchone()
            if row is None:
                return None
            plan = dict(row)
            plan["tasks"] = [
                dict(task)
                for task in conn.execute(
                    "select * from plan_tasks where plan_id = ? order by task_id", (plan_id,)
                ).fetchall()
            ]
        return plan

    def list_events(
        self,
        plan_id: str,
        *,
        task_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = "select * from plan_events where plan_id = ?"
        params: list[Any] = [plan_id]
        if task_id:
            query += " and task_id = ?"
            params.append(task_id)
        query += " order by id asc limit ?"
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            with suppress(json.JSONDecodeError):
                item["payload"] = json.loads(item.get("payload") or "{}")
            events.append(item)
        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _upsert_task(
        self, conn: sqlite3.Connection, plan_id: str, task: Task, *, tokens: int = 0, turns: int = 0
    ) -> None:
        conn.execute(
            """
            insert or replace into plan_tasks
                (plan_id, task_id, description, type, resources, dependencies, status,
                 attempts, error, result_preview, tokens, turns, started_at, ended_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                task.id,
                task.description,
                task.type.value,
                ",".join(task.resources),
                ",".join(task.dependencies),
                task.status.value,
                task.attempts,
                task.error,
                task.result[:500],
                tokens,
                turns,
                _iso(task.start_time),
                _iso(task.end_time),
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, isolation_level=None)


def _iso(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _compact(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:497] + "..."
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth + 1)
            for key, item in value.items()
            if key not in {"messages", "task_description"}
        }
    if isinstance(value, list):
        return [_compact(item, depth + 1) for item in value[:20]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:200]
