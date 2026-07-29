from uuid import uuid4
from app.models import TaskPlan, PlanNode, NodeType, NodeStatus, ExecutionBudget, WorkerContext
from app.scheduler import PlanScheduler

def test():
    plan = TaskPlan(
        task_id=uuid4(),
        goal="Test goal string long enough",
        budget=ExecutionBudget(max_retries=1, max_nodes=10),
        nodes=[
            PlanNode(id="research", type=NodeType.RESEARCH, objective="research task", depends_on=[]),
            PlanNode(id="analyze", type=NodeType.ANALYZE, objective="analyze task", depends_on=["research"]),
            PlanNode(id="draft", type=NodeType.DRAFT, objective="draft task", depends_on=["analyze"]),
            PlanNode(id="verify", type=NodeType.VERIFY, objective="verify task", depends_on=["draft"]),
        ]
    )

    context = WorkerContext(
        task_id=plan.task_id,
        objective="Test",
        budget=plan.budget
    )

    while True:
        plan = PlanScheduler.refresh_ready(plan)
        ready_nodes = [n for n in plan.nodes if n.status == NodeStatus.READY]
        print(f"Ready nodes: {[n.id for n in ready_nodes]}")
        
        if not ready_nodes:
            break

        for node in ready_nodes:
            plan = PlanScheduler.start(plan, node.id)
            print(f"Running node: {node.id}")
            
            # Simulate worker output
            if node.type == NodeType.VERIFY:
                verification_passed = False
                
                if not verification_passed and context.budget.max_retries > 0:
                    context.budget.max_retries -= 1
                    draft_node_id = node.depends_on[0] if node.depends_on else None
                    draft_node = next((n for n in plan.nodes if n.id == draft_node_id), None) if draft_node_id else None
                    
                    if draft_node:
                        retry_idx = context.budget.max_retries
                        new_draft_id = f"retry_draft_{retry_idx}"
                        new_verify_id = f"retry_verify_{retry_idx}"
                        
                        plan.nodes.append(PlanNode(
                            id=new_draft_id,
                            type=NodeType.DRAFT,
                            objective="Revise",
                            depends_on=draft_node.depends_on.copy(),
                            status=NodeStatus.PENDING
                        ))
                        plan.nodes.append(PlanNode(
                            id=new_verify_id,
                            type=NodeType.VERIFY,
                            objective="Verify",
                            depends_on=[new_draft_id],
                            status=NodeStatus.PENDING
                        ))
                        print(f"Appended {new_draft_id} and {new_verify_id}")
            
            plan = PlanScheduler.complete(plan, node.id, succeeded=True)

    print("Final nodes:")
    for n in plan.nodes:
        print(f"- {n.id}: {n.status}")

if __name__ == "__main__":
    test()
