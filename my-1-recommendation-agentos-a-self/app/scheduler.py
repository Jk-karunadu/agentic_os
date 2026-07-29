from app.models import NodeStatus, TaskPlan


class PlanTransitionError(ValueError):
    """Raised when a caller attempts an invalid task-graph state transition."""


class PlanScheduler:
    """Deterministic DAG scheduler; worker execution is deliberately separate."""

    @staticmethod
    def refresh_ready(plan: TaskPlan) -> TaskPlan:
        updated = plan.model_copy(deep=True)
        completed = {node.id for node in updated.nodes if node.status == NodeStatus.COMPLETED}
        for node in updated.nodes:
            if node.status == NodeStatus.PENDING and set(node.depends_on).issubset(completed):
                node.status = NodeStatus.READY
        return updated

    @classmethod
    def ready_nodes(cls, plan: TaskPlan) -> list[str]:
        refreshed = cls.refresh_ready(plan)
        return [node.id for node in refreshed.nodes if node.status == NodeStatus.READY]

    @classmethod
    def start(cls, plan: TaskPlan, node_id: str) -> TaskPlan:
        updated = cls.refresh_ready(plan)
        node = cls._find(updated, node_id)
        if node.status != NodeStatus.READY:
            raise PlanTransitionError(f"Node '{node_id}' is not ready to run (status: {node.status}).")
        node.status = NodeStatus.RUNNING
        return updated

    @classmethod
    def complete(cls, plan: TaskPlan, node_id: str, succeeded: bool) -> TaskPlan:
        updated = plan.model_copy(deep=True)
        node = cls._find(updated, node_id)
        if node.status != NodeStatus.RUNNING:
            raise PlanTransitionError(f"Node '{node_id}' is not running (status: {node.status}).")
        node.status = NodeStatus.COMPLETED if succeeded else NodeStatus.FAILED
        return cls.refresh_ready(updated) if succeeded else updated

    @staticmethod
    def _find(plan: TaskPlan, node_id: str):
        for node in plan.nodes:
            if node.id == node_id:
                return node
        raise PlanTransitionError(f"Unknown node '{node_id}'.")
