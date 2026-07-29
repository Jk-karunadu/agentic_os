import requests
import time
from typing import Any

BASE_URL = "http://127.0.0.1:8000"

def create_task(objective: str, evidence: list[dict]) -> dict:
    """Creates a new task and returns the task data (including task_id)."""
    response = requests.post(
        f"{BASE_URL}/tasks",
        json={"objective": objective, "evidence": evidence},
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def run_orchestrated(task_id: str, status_callback: Any = None) -> dict:
    """Runs the orchestrated multi-agent DAG for a given task using background polling."""
    # 1. Start the background task
    response = requests.post(
        f"{BASE_URL}/tasks/{task_id}/run-orchestrated",
        timeout=10
    )
    response.raise_for_status()
    
    # 2. Poll until it finishes
    while True:
        status_response = requests.get(
            f"{BASE_URL}/tasks/{task_id}/run-orchestrated/status",
            timeout=10
        )
        status_response.raise_for_status()
        data = status_response.json()
        
        if data.get("status") == "running":
            if status_callback and data.get("plan"):
                status_callback(data["plan"])
            time.sleep(2)
            continue
            
        return data

def search_memory(query: str, type_filter: str | None = None) -> list[dict]:
    """Queries the semantic memory database via POST /memory/query."""
    payload: dict[str, Any] = {"query": query, "top_k": 5}
    if type_filter:
        payload["memory_type"] = type_filter

    response = requests.post(
        f"{BASE_URL}/memory/query",
        json=payload,
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def get_failures() -> list[dict]:
    """Retrieves all logged failures from the failure database."""
    response = requests.get(
        f"{BASE_URL}/failures",
        timeout=10
    )
    response.raise_for_status()
    return response.json()

def check_health() -> dict:
    """Check the API and Ollama health status."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"api": False, "ollama": False}
