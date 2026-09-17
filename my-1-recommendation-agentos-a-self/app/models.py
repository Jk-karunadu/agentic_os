from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Evidence(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl
    excerpt: str = Field(min_length=1, max_length=12_000)
    source_type: str = Field(default="web", max_length=40)


class TaskCreate(BaseModel):
    objective: str = Field(min_length=10, max_length=2_000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=12)


class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    objective: str
    evidence: list[Evidence] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunTrace(BaseModel):
    task_id: UUID
    model: str
    status: TaskStatus
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    error: str | None = None


class TaskResult(BaseModel):
    task_id: UUID
    answer: str
    trace: RunTrace


class NodeType(StrEnum):
    RESEARCH = "research"
    ANALYZE = "analyze"
    DRAFT = "draft"
    VERIFY = "verify"


class NodeStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionBudget(BaseModel):
    max_nodes: int = Field(default=12, ge=1, le=20)
    max_tool_calls: int = Field(default=12, ge=0, le=30)
    max_retries: int = Field(default=1, ge=0, le=2)
    max_output_tokens: int = Field(default=900, ge=100, le=2_000)


class NodeEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    role: str
    content: str
    name: str | None = None


class PlanNode(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,30}$")
    type: NodeType
    objective: str = Field(min_length=5, max_length=600)
    depends_on: list[str] = Field(default_factory=list, max_length=24)
    status: NodeStatus = NodeStatus.PENDING
    events: list[NodeEvent] = Field(default_factory=list)


class TaskPlan(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    version: int = Field(default=1, ge=1)
    goal: str = Field(min_length=10, max_length=2_000)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    nodes: list[PlanNode] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskPlan":
        if len(self.nodes) > self.budget.max_nodes:
            raise ValueError("Plan exceeds the configured node budget.")
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Plan node IDs must be unique.")
        for node in self.nodes:
            unknown = set(node.depends_on) - node_ids
            if unknown:
                raise ValueError(f"Node '{node.id}' has unknown dependencies: {sorted(unknown)}")
            if node.id in node.depends_on:
                raise ValueError(f"Node '{node.id}' cannot depend on itself.")

        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {node.id: node.depends_on for node in self.nodes}

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("Plan graph must not contain cycles.")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in graph[node_id]:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in graph:
            visit(node_id)
        return self


class PlanDraft(BaseModel):
    """LLM-facing plan schema; runtime-owned identifiers are added after validation."""

    goal: str = Field(min_length=10, max_length=2_000)
    nodes: list[PlanNode] = Field(min_length=1, max_length=12)


class ReplanRequest(BaseModel):
    failed_node_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,30}$")
    failure_reason: str = Field(min_length=5, max_length=1_000)


class NodeCompletion(BaseModel):
    succeeded: bool


class IssueSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationIssue(BaseModel):
    type: str = Field(min_length=3, max_length=80)
    severity: IssueSeverity
    claim: str = Field(min_length=1, max_length=1_000)
    evidence: str = Field(min_length=1, max_length=2_000)
    repair_instruction: str = Field(min_length=1, max_length=1_000)


class VerificationResult(BaseModel):
    task_id: UUID
    answer: str = Field(min_length=1, max_length=20_000)
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    verifier: str


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=20_000)


class ReflectionLesson(BaseModel):
    task_id: UUID
    trigger: str = Field(min_length=3, max_length=160)
    lesson: str = Field(min_length=10, max_length=1_000)
    recommended_action: str = Field(min_length=5, max_length=1_000)
    scope: str = Field(default="research_synthesis", max_length=100)
    confidence: float = Field(ge=0, le=1)


class ErrorType(StrEnum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_MISMATCH = "citation_mismatch"
    PLANNING_ERROR = "planning_error"
    SOURCE_QUALITY_FAILURE = "source_quality_failure"
    TOOL_SCHEMA_ERROR = "tool_schema_error"
    MISSING_SECTION = "missing_section"
    CONTENT_TOO_SHORT = "content_too_short"


class FailureRecord(BaseModel):
    id: int | None = None
    task_id: UUID
    plan_version: int | None = None
    failing_node: str | None = None
    error_type: ErrorType
    error_detail: str = Field(min_length=5, max_length=2_000)
    verification_json: str | None = None
    lesson_json: str | None = None
    repair_attempted: bool = False
    repair_succeeded: bool | None = None
    model: str | None = None


class VerifiedRunResult(BaseModel):
    """Composite result from the run-verified endpoint."""
    task_id: UUID
    answer: str
    verification: VerificationResult
    lesson: ReflectionLesson | None = None
    failure: FailureRecord | None = None
    retried: bool = False
    retry_answer: str | None = None
    retry_verification: VerificationResult | None = None


# ── Phase 4: Memory system models ───────────────────────────────────

class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: MemoryType
    text: str = Field(min_length=1, max_length=8_000)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_task_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success_score: float = Field(default=0.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata_json: str = Field(default="{}")


class MemoryQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    memory_type: MemoryType | None = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class MemoryScoreBreakdown(BaseModel):
    """Explains why a memory item was retrieved and its rank."""
    memory_id: str
    text: str
    total_score: float
    similarity: float
    importance: float
    recency: float
    success: float
    type_match: float


class EntityRelation(BaseModel):
    id: int | None = None
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=200)
    object: str = Field(min_length=1, max_length=300)
    source_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


# ── Phase 5: Multi-agent workers ────────────────────────────────────

class WorkerContext(BaseModel):
    task_id: UUID
    objective: str
    evidence: list[Evidence] = Field(default_factory=list)
    previous_results: dict[str, str] = Field(default_factory=dict)
    lessons: list[str] = Field(default_factory=list)
    budget: ExecutionBudget


class WorkerResult(BaseModel):
    node_id: str
    output: str
    evidence_used: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    duration_ms: int = 0
    error: str | None = None


class OrchestratedRunResult(BaseModel):
    task_id: UUID
    final_answer: str | None = None
    verification_passed: bool = False
    verification_output: str | None = None
    node_results: dict[str, WorkerResult] = Field(default_factory=dict)
    total_tokens: int = 0
    total_duration_ms: int = 0


# ── Chat models ──────────────────────────────────────────────────────

class ChatMessageModel(BaseModel):
    role: str
    content: str
    tools_used: list[str] = []
    new_tools: list[str] = []
    created_at: str = ""


class ChatRequest(BaseModel):
    prompt: str
    session_id: str = ""


class ChatResponse(BaseModel):
    content: str
    session_id: str
    tools_used: list[str] = []
    new_tools_created: list[str] = []
    duration_ms: int = 0
    timed_out: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
