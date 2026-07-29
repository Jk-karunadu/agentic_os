from abc import ABC, abstractmethod

from app.local_llm import OllamaClient
from app.models import NodeType, PlanNode, WorkerContext, WorkerResult


class BaseWorker(ABC):
    """Abstract base class for specialized multi-agent workers."""

    name: str
    allowed_node_types: set[NodeType]

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    @abstractmethod
    def execute(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        """Execute the worker's specialized task."""
        pass
