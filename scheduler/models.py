"""Scheduler data models."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from workflow.definition import WorkflowDefinition


class ScheduleTriggerType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"


class ScheduleStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class ScheduleRecord(BaseModel):
    schedule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    workflow_definition: WorkflowDefinition
    trigger_type: ScheduleTriggerType
    # Trigger kwargs forwarded directly to APScheduler:
    #   cron    → {"hour": 9, "minute": 0, "day_of_week": "mon-fri"}
    #   interval → {"seconds": 30}  /  {"minutes": 5}
    trigger_config: dict
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_at: datetime | None = None
