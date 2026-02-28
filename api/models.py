"""API request and response models."""

from pydantic import BaseModel

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
