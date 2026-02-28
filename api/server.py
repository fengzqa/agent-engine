"""FastAPI service layer for Agent Engine."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.models import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
    RunWorkflowRequest,
    RunWorkflowResponse,
)
from core.agent import Agent
from core.state import WorkflowInstance, WorkflowStatus
from store.workflow_store import WorkflowStore
from tools.builtin.common import HttpRequestTool, ReadFileTool, RunPythonTool, WriteFileTool
from tools.registry import ToolRegistry
from workflow.runner import WorkflowRunner

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
# Tools are registered at module load time so they're available even when
# ASGITransport doesn't trigger the lifespan (e.g. in tests).

_store = WorkflowStore()
_tool_registry = ToolRegistry()
_tool_registry.register(HttpRequestTool())
_tool_registry.register(RunPythonTool())
_tool_registry.register(ReadFileTool())
_tool_registry.register(WriteFileTool())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _store.init()   # initialises DB tables on startup
    yield


app = FastAPI(
    title="Agent Engine API",
    description="Workflow automation powered by Claude.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Background helpers ────────────────────────────────────────────────────────

async def _run_workflow_bg(definition, instance_id: str) -> None:
    runner = WorkflowRunner(_tool_registry, store=_store)
    try:
        await runner.run(definition, instance_id=instance_id)
    except Exception as e:
        logger.error("Background workflow error: %s", e)


async def _resume_workflow_bg(instance_id: str, definition, retry_failed: bool) -> None:
    runner = WorkflowRunner(_tool_registry, store=_store)
    try:
        await runner.resume(instance_id, definition, retry_failed=retry_failed)
    except Exception as e:
        logger.error("Background resume error: %s", e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/workflows/run", response_model=RunWorkflowResponse, status_code=202)
async def run_workflow(req: RunWorkflowRequest):
    """Submit a workflow for async execution. Returns instance_id immediately."""
    # Pre-save a PENDING record so clients can poll right away
    instance = WorkflowInstance(
        workflow_name=req.definition.name,
        status=WorkflowStatus.PENDING,
        steps={},
    )
    await _store.save(instance)
    asyncio.create_task(_run_workflow_bg(req.definition, instance.instance_id))
    return RunWorkflowResponse(instance_id=instance.instance_id, status="pending")


@app.get("/workflows")
async def list_workflows(workflow_name: str | None = None):
    """List workflow instances, optionally filtered by name."""
    return await _store.list_all(workflow_name=workflow_name)


@app.get("/workflows/{instance_id}")
async def get_workflow(instance_id: str):
    """Get current state of a workflow instance."""
    try:
        instance = await _store.load(instance_id)
    except KeyError:
        raise HTTPException(404, detail=f"Instance '{instance_id}' not found")
    return instance.model_dump()


@app.post("/workflows/{instance_id}/resume", response_model=RunWorkflowResponse, status_code=202)
async def resume_workflow(instance_id: str, req: ResumeRequest):
    """Resume a failed or interrupted workflow."""
    try:
        await _store.load(instance_id)
    except KeyError:
        raise HTTPException(404, detail=f"Instance '{instance_id}' not found")
    asyncio.create_task(_resume_workflow_bg(instance_id, req.definition, req.retry_failed))
    return RunWorkflowResponse(instance_id=instance_id, status="resuming")


@app.delete("/workflows/{instance_id}", status_code=204)
async def delete_workflow(instance_id: str):
    """Delete a workflow instance."""
    try:
        await _store.load(instance_id)
    except KeyError:
        raise HTTPException(404, detail=f"Instance '{instance_id}' not found")
    await _store.delete(instance_id)


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """Single-turn agent chat with tool access."""
    agent = Agent(registry=_tool_registry, system_prompt=req.system_prompt)
    response = await agent.run(req.message)
    return ChatResponse(response=response)
