import json
import sqlite3
from pathlib import Path
from uuid import UUID

from app.models import FailureRecord, ReflectionLesson, RunTrace, Task, TaskPlan, TaskStatus, VerificationResult


class TaskStore:
    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    lesson_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    output_tokens INTEGER,
                    duration_ms INTEGER,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    plan_version INTEGER,
                    failing_node TEXT,
                    error_type TEXT NOT NULL,
                    error_detail TEXT NOT NULL,
                    verification_json TEXT,
                    lesson_json TEXT,
                    repair_attempted BOOLEAN DEFAULT 0,
                    repair_succeeded BOOLEAN,
                    model TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def create(self, task: Task) -> Task:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(task.id), task.objective,
                    json.dumps([item.model_dump(mode="json") for item in task.evidence]),
                    task.status.value, task.created_at.isoformat(), task.updated_at.isoformat(),
                ),
            )
        return task

    def get(self, task_id: UUID) -> Task | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (str(task_id),)).fetchone()
        if row is None:
            return None
        return Task.model_validate(
            {
                "id": row["id"], "objective": row["objective"],
                "evidence": json.loads(row["evidence_json"]), "status": row["status"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
            }
        )

    def set_status(self, task_id: UUID, status: TaskStatus) -> None:
        from datetime import UTC, datetime

        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, datetime.now(UTC).isoformat(), str(task_id)),
            )

    def add_trace(self, trace: RunTrace) -> None:
        from datetime import UTC, datetime

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_traces
                (task_id, model, status, prompt_tokens, output_tokens, duration_ms, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trace.task_id), trace.model, trace.status.value, trace.prompt_tokens,
                    trace.output_tokens, trace.duration_ms, trace.error or "",
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list_traces(self, task_id: UUID) -> list[RunTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_traces WHERE task_id = ? ORDER BY id", (str(task_id),)
            ).fetchall()
        return [
            RunTrace(
                task_id=row["task_id"], model=row["model"], status=row["status"],
                prompt_tokens=row["prompt_tokens"], output_tokens=row["output_tokens"],
                duration_ms=row["duration_ms"], error=row["error"] or None,
            )
            for row in rows
        ]

    def save_plan(self, plan: TaskPlan) -> TaskPlan:
        from datetime import UTC, datetime

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO plans VALUES (?, ?, ?, ?, ?)",
                (
                    str(plan.id), str(plan.task_id), plan.version,
                    plan.model_dump_json(), datetime.now(UTC).isoformat(),
                ),
            )
        return plan

    def latest_plan(self, task_id: UUID) -> TaskPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM plans WHERE task_id = ? ORDER BY version DESC LIMIT 1",
                (str(task_id),),
            ).fetchone()
        return TaskPlan.model_validate_json(row["plan_json"]) if row else None

    def update_plan(self, plan: TaskPlan) -> TaskPlan:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE plans SET plan_json = ? WHERE id = ?",
                (plan.model_dump_json(), str(plan.id)),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Plan '{plan.id}' does not exist.")
        return plan

    def add_verification(self, result: VerificationResult) -> VerificationResult:
        from datetime import UTC, datetime
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO verifications (task_id, result_json, created_at) VALUES (?, ?, ?)",
                (str(result.task_id), result.model_dump_json(), datetime.now(UTC).isoformat()),
            )
        return result

    def latest_verification(self, task_id: UUID) -> VerificationResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM verifications WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (str(task_id),),
            ).fetchone()
        return VerificationResult.model_validate_json(row["result_json"]) if row else None

    def add_lesson(self, lesson: ReflectionLesson) -> ReflectionLesson:
        from datetime import UTC, datetime
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO lessons (task_id, lesson_json, created_at) VALUES (?, ?, ?)",
                (str(lesson.task_id), lesson.model_dump_json(), datetime.now(UTC).isoformat()),
            )
        return lesson

    def list_lessons(self, task_id: UUID | None = None) -> list[ReflectionLesson]:
        with self._connect() as connection:
            if task_id:
                rows = connection.execute(
                    "SELECT lesson_json FROM lessons WHERE task_id = ? ORDER BY id", (str(task_id),)
                ).fetchall()
            else:
                rows = connection.execute("SELECT lesson_json FROM lessons ORDER BY id").fetchall()
        return [ReflectionLesson.model_validate_json(row["lesson_json"]) for row in rows]

    # ── Failure database ─────────────────────────────────────────────

    def add_failure(self, failure: FailureRecord) -> FailureRecord:
        from datetime import UTC, datetime
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO failures
                (task_id, plan_version, failing_node, error_type, error_detail,
                 verification_json, lesson_json, repair_attempted, repair_succeeded, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(failure.task_id), failure.plan_version, failure.failing_node,
                    failure.error_type.value, failure.error_detail,
                    failure.verification_json, failure.lesson_json,
                    failure.repair_attempted, failure.repair_succeeded,
                    failure.model, datetime.now(UTC).isoformat(),
                ),
            )
            failure.id = cursor.lastrowid
        return failure

    def list_failures(self, task_id: UUID | None = None) -> list[FailureRecord]:
        with self._connect() as connection:
            if task_id:
                rows = connection.execute(
                    "SELECT * FROM failures WHERE task_id = ? ORDER BY id", (str(task_id),)
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM failures ORDER BY id").fetchall()
        return [self._row_to_failure(row) for row in rows]

    def get_failures_by_type(self, error_type: str) -> list[FailureRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM failures WHERE error_type = ? ORDER BY id", (error_type,)
            ).fetchall()
        return [self._row_to_failure(row) for row in rows]

    def update_failure_repair(self, failure_id: int, succeeded: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE failures SET repair_attempted = 1, repair_succeeded = ? WHERE id = ?",
                (succeeded, failure_id),
            )

    @staticmethod
    def _row_to_failure(row: sqlite3.Row) -> FailureRecord:
        return FailureRecord(
            id=row["id"], task_id=row["task_id"],
            plan_version=row["plan_version"], failing_node=row["failing_node"],
            error_type=row["error_type"], error_detail=row["error_detail"],
            verification_json=row["verification_json"], lesson_json=row["lesson_json"],
            repair_attempted=bool(row["repair_attempted"]),
            repair_succeeded=bool(row["repair_succeeded"]) if row["repair_succeeded"] is not None else None,
            model=row["model"],
        )
