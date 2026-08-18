from datetime import UTC
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from faulttrace_core.annotations import AnnotationAssignment, AnnotationStatus, AnnotationTask

router = APIRouter(prefix="/annotations", tags=["annotations"])

# In-memory mock for the pilot. Real implementation would use the database.
_MOCK_TASKS: dict[str, AnnotationTask] = {}
_MOCK_ASSIGNMENTS: dict[str, AnnotationAssignment] = {}


@router.get("/tasks", response_model=list[AnnotationTask])
async def list_tasks():
    return list(_MOCK_TASKS.values())


@router.post("/tasks", response_model=AnnotationTask)
async def create_task(world_id: str, query_id: str, context_payload: dict[str, Any]):
    task_id = f"task_{uuid4().hex[:8]}"
    task = AnnotationTask(
        task_id=task_id, world_id=world_id, query_id=query_id, context_payload=context_payload
    )
    _MOCK_TASKS[task_id] = task
    return task


@router.post("/assignments", response_model=AnnotationAssignment)
async def assign_task(task_id: str, user_id: str):
    if task_id not in _MOCK_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    assignment_id = f"asn_{uuid4().hex[:8]}"
    assignment = AnnotationAssignment(assignment_id=assignment_id, task_id=task_id, user_id=user_id)
    _MOCK_ASSIGNMENTS[assignment_id] = assignment
    return assignment


@router.post("/assignments/{assignment_id}/submit", response_model=AnnotationAssignment)
async def submit_annotation(assignment_id: str, result: dict[str, Any], time_spent: int = 0):
    if assignment_id not in _MOCK_ASSIGNMENTS:
        raise HTTPException(status_code=404, detail="Assignment not found")

    assignment = _MOCK_ASSIGNMENTS[assignment_id]
    assignment.result = result
    assignment.time_spent_seconds = time_spent
    from datetime import datetime

    assignment.completed_at = datetime.now(UTC)

    # Update task status if needed
    task = _MOCK_TASKS.get(assignment.task_id)
    if task:
        task.status = AnnotationStatus.IN_PROGRESS

    return assignment
