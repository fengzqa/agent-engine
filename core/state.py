"""Workflow state machine types."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepState(BaseModel):
    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class WorkflowInstance(BaseModel):
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: dict[str, StepState] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.created_at and self.finished_at:
            return (self.finished_at - self.created_at).total_seconds()
        return None

    def summary(self) -> str:
        lines = [
            f"Workflow : {self.workflow_name}",
            f"Status   : {self.status.value}",
            f"Duration : {self.duration_seconds:.2f}s" if self.duration_seconds else "Duration : -",
            "",
            f"{'Step':<20} {'Status':<12} {'Duration':>10}",
            "-" * 46,
        ]
        for state in self.steps.values():
            dur = f"{state.duration_seconds:.2f}s" if state.duration_seconds else "-"
            lines.append(f"{state.step_id:<20} {state.status.value:<12} {dur:>10}")
        return "\n".join(lines)
