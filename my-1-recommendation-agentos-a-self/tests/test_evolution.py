import pytest
from uuid import uuid4
from app.models import ReflectionLesson, TaskPlan, PlanNode, NodeType, NodeStatus, ExecutionBudget
from app.store import TaskStore
from app.memory import MemoryStore
from app.tools.base import ToolRegistry, BaseTool
from app.tools.tool_generator import ToolGeneratorTool


def test_lesson_credit_assignment(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = TaskStore(db_path)

    lesson = ReflectionLesson(
        task_id=uuid4(),
        trigger="missing_citation",
        lesson="Always cite evidence URLs.",
        recommended_action="Attach exact URL to claims.",
        confidence=0.8,
        effectiveness_score=0.5,
    )
    saved = store.add_lesson(lesson)
    assert saved.id is not None
    assert saved.effectiveness_score == 0.5
    assert saved.success_count == 0

    # Reward lesson
    rewarded = store.reward_lesson(saved.id, delta=0.15)
    assert rewarded is not None
    assert rewarded.success_count == 1
    assert rewarded.effectiveness_score == 0.65

    # Penalize lesson
    penalized = store.penalize_lesson(saved.id, delta=0.25)
    assert penalized is not None
    assert penalized.failure_count == 1
    assert penalized.effectiveness_score == 0.40


def test_lesson_pruning(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = TaskStore(db_path)

    # Effective lesson
    l1 = ReflectionLesson(
        task_id=uuid4(), trigger="trigger_valid_1", lesson="Lesson description 1", recommended_action="Action to take 1",
        confidence=0.8, effectiveness_score=0.8, success_count=3, failure_count=0,
    )
    store.add_lesson(l1)

    # Ineffective lesson
    l2 = ReflectionLesson(
        task_id=uuid4(), trigger="trigger_valid_2", lesson="Lesson description 2", recommended_action="Action to take 2",
        confidence=0.4, effectiveness_score=0.1, success_count=0, failure_count=3,
    )
    store.add_lesson(l2)

    assert len(store.list_lessons()) == 2
    pruned = store.prune_ineffective_lessons(threshold=0.20)
    assert pruned == 1
    remaining = store.list_lessons()
    assert len(remaining) == 1
    assert remaining[0].trigger == "trigger_valid_1"


def test_memory_procedural_deduplication_and_retrieval(tmp_path):
    db_path = str(tmp_path / "test.db")
    mem_store = MemoryStore(db_path)

    # Store first lesson
    m1 = mem_store.store_procedural_lesson(
        task_id=str(uuid4()),
        trigger="citation_mismatch",
        lesson_text="Do not cite URLs not in evidence.",
        recommended_action="Remove foreign URL citations.",
        confidence=0.85,
    )
    assert m1.id is not None

    # Retrieve relevant lesson
    results = mem_store.retrieve_relevant_lessons("Check citations and URLs in research output", top_k=2)
    assert len(results) >= 1
    assert "citation_mismatch" in results[0].text

    # Store nearly duplicate lesson - should merge/deduplicate rather than create a new row
    m2 = mem_store.store_procedural_lesson(
        task_id=str(uuid4()),
        trigger="citation_mismatch",
        lesson_text="Do not cite URLs not in evidence.",
        recommended_action="Remove foreign URL citations.",
        confidence=0.90,
    )
    assert m2.id == m1.id  # Same memory id because it deduplicated!


def test_exemplar_memory(tmp_path):
    db_path = str(tmp_path / "test.db")
    mem_store = MemoryStore(db_path)

    mem_store.store_exemplar(
        task_id=str(uuid4()),
        objective="Summarise AI safety techniques",
        plan_summary="research -> draft -> verify",
        final_answer="RLHF and Constitutional AI are two key techniques.",
        tokens_used=150,
    )

    exemplars = mem_store.retrieve_exemplars("Explain AI safety approaches", top_k=1)
    assert len(exemplars) == 1
    assert "RLHF and Constitutional AI" in exemplars[0].text


def test_tool_self_healing():
    registry = ToolRegistry()

    class BrokenTool(BaseTool):
        name = "flaky_tool"
        description = "A tool that throws an error initially"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.fail = True

        def execute(self, **kwargs) -> str:
            if self.fail:
                raise ValueError("Division by zero in calculation")
            return "Calculation succeeded: 42"

    tool = BrokenTool()
    registry.register(tool)

    # Mock healer
    class MockHealer:
        def heal_tool(self, name: str, runtime_error: str) -> bool:
            if name == "flaky_tool" and "Division by zero" in runtime_error:
                # Fix the tool
                fixed_tool = BrokenTool()
                fixed_tool.fail = False
                registry.register(fixed_tool)
                return True
            return False

    registry.set_healer(MockHealer())

    # Execute should trigger healer and return the self-healed result
    res = registry.execute_tool("flaky_tool", {})
    assert "[Self-Healed Tool 'flaky_tool']" in res
    assert "42" in res
