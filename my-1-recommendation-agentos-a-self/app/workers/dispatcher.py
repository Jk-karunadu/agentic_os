from app.local_llm import OllamaClient
from app.models import NodeType, PlanNode, WorkerContext, WorkerResult
from app.tools.base import ToolRegistry
from app.workers.analyst import AnalystWorker
from app.workers.base import BaseWorker
from app.workers.research import ResearchWorker
from app.workers.verifier import VerifierWorker
from app.workers.writer import WriterWorker


class WorkerDispatcher:
    """Dispatches execution to the appropriate specialized worker based on NodeType."""

    def __init__(self, llm_client: OllamaClient, tool_registry: ToolRegistry | None = None) -> None:
        self.workers: dict[NodeType, BaseWorker] = {
            NodeType.RESEARCH: ResearchWorker(llm_client, tool_registry),
            NodeType.ANALYZE: AnalystWorker(llm_client),
            NodeType.DRAFT: WriterWorker(llm_client),
            NodeType.VERIFY: VerifierWorker(llm_client),
        }

    def execute_node(self, node: PlanNode, context: WorkerContext, log_event=None) -> WorkerResult:
        """Find the correct worker for the node and execute it."""
        worker = self.workers.get(node.type)
        if not worker:
            return WorkerResult(
                node_id=node.id,
                output="",
                error=f"No worker available for node type '{node.type}'",
            )
        
        try:
            return worker.execute(node, context, log_event)
        except Exception as e:
            return WorkerResult(
                node_id=node.id,
                output="",
                error=f"Worker '{worker.name}' encountered an error: {e}",
            )
