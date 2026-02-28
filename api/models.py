"""API request and response models."""

from datetime import datetime

from pydantic import BaseModel

from scheduler.models import ScheduleTriggerType
from workflow.definition import WorkflowDefinition


class RunWorkflowRequest(BaseModel):
    definition: WorkflowDefinition


class RunWorkflowResponse(BaseModel):
    instance_id: str
    status: str


class ResumeRequest(BaseModel):
    definition: WorkflowDefinition
    retry_failed: bool = True


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful workflow automation assistant."


class ChatResponse(BaseModel):
    response: str


class CreateScheduleRequest(BaseModel):
    name: str
    definition: WorkflowDefinition
    trigger: ScheduleTriggerType
    trigger_config: dict


class ScheduleResponse(BaseModel):
    schedule_id: str
    name: str
    trigger: str
    trigger_config: dict
    status: str
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    created_at: datetime
