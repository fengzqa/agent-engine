"""FastAPI service layer for Agent Engine."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    ChatRequest,
    ChatResponse,
    CreateScheduleRequest,
    ResumeRequest,
    RunWorkflowRequest,
    RunWorkflowResponse,
    ScheduleResponse,
)
from core.agent import Agent
from core.event_bus import EventBus
from core.state import WorkflowInstance, WorkflowStatus
from scheduler.models import ScheduleRecord
from scheduler.schedule_store import ScheduleStore
from scheduler.workflow_scheduler import WorkflowScheduler
from store.workflow_store import WorkflowStore
from tools.builtin.common import HttpRequestTool, ReadFileTool, RunPythonTool, WriteFileTool
from tools.registry import ToolRegistry
from workflow.runner import WorkflowRunner, _make_event

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
# Tools are registered at module load time so they're available even when
# ASGITransport doesn't trigger the lifespan (e.g. in tests).

_store = WorkflowStore()
_schedule_store = ScheduleStore()
_event_bus = EventBus()
_tool_registry = ToolRegistry()
_tool_registry.register(HttpRequestTool())
_tool_registry.register(RunPythonTool())
_tool_registry.register(ReadFileTool())
_tool_registry.register(WriteFileTool())

_runner = WorkflowRunner(_tool_registry, store=_store, event_bus=_event_bus)
_scheduler = WorkflowScheduler(_runner, _schedule_store, _store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _store.init()
    await _schedule_store.init()
    await _scheduler.start()
    yield
    await _scheduler.shutdown()


app = FastAPI(
    title="Agent Engine API",
    description="Workflow automation powered by Claude.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Background helpers ────────────────────────────────────────────────────────

async def _run_workflow_bg(definition, instance_id: str) -> None:
    runner = WorkflowRunner(_tool_registry, store=_store, event_bus=_event_bus)
    try:
        await runner.run(definition, instance_id=instance_id)
    except Exception as e:
        logger.error("Background workflow error: %s", e)


async def _resume_workflow_bg(instance_id: str, definition, retry_failed: bool) -> None:
    runner = WorkflowRunner(_tool_registry, store=_store, event_bus=_event_bus)
    try:
        await runner.resume(instance_id, definition, retry_failed=retry_failed)
    except Exception as e:
        logger.error("Background resume error: %s", e)


# ── Schedule helpers ──────────────────────────────────────────────────────────

def _schedule_response(record: ScheduleRecord) -> ScheduleResponse:
    return ScheduleResponse(
        schedule_id=record.schedule_id,
        name=record.name,
        trigger=record.trigger_type.value,
        trigger_config=record.trigger_config,
        status=record.status.value,
        next_run_at=_scheduler.next_run_time(record.schedule_id),
        last_run_at=record.last_run_at,
        created_at=record.created_at,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/workflows/run", response_model=RunWorkflowResponse, status_code=202)
async def run_workflow(req: RunWorkflowRequest):
    """Submit a workflow for async execution. Returns instance_id immediately."""
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


_SSE_TERMINAL = {"completed", "failed"}
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.get("/workflows/{instance_id}/stream")
async def stream_workflow(instance_id: str):
    """Stream live workflow state changes as Server-Sent Events.

    Immediately sends the current snapshot, then pushes every subsequent
    state change until the workflow reaches a terminal status or the client
    disconnects.  Each event is a JSON-encoded state dict on a ``data:`` line.
    A comment line (``: heartbeat``) is sent every 30 s to keep the connection alive.
    """
    try:
        instance = await _store.load(instance_id)
    except KeyError:
        raise HTTPException(404, detail=f"Instance '{instance_id}' not found")

    async def generator():
        # ── 1. Send current snapshot ──────────────────────────────────────
        snapshot = _make_event(instance)
        yield f"data: {json.dumps(snapshot)}\n\n"
        if snapshot["status"] in _SSE_TERMINAL:
            return

        # ── 2. Subscribe and forward subsequent events ────────────────────
        q = _event_bus.subscribe(instance_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in _SSE_TERMINAL:
                        return
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _event_bus.unsubscribe(instance_id, q)

    return StreamingResponse(generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """Single-turn agent chat with tool access."""
    agent = Agent(registry=_tool_registry, system_prompt=req.system_prompt)
    response = await agent.run(req.message)
    return ChatResponse(response=response)


# ── Schedule routes ───────────────────────────────────────────────────────────

@app.post("/schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(req: CreateScheduleRequest):
    """Create a new cron or interval schedule."""
    record = ScheduleRecord(
        name=req.name,
        workflow_definition=req.definition,
        trigger_type=req.trigger,
        trigger_config=req.trigger_config,
    )
    await _scheduler.add_schedule(record)
    return _schedule_response(record)


@app.get("/schedules")
async def list_schedules():
    """List all schedules."""
    return await _schedule_store.list_all()


@app.get("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(schedule_id: str):
    """Get a schedule by ID."""
    try:
        record = await _schedule_store.load(schedule_id)
    except KeyError:
        raise HTTPException(404, detail=f"Schedule '{schedule_id}' not found")
    return _schedule_response(record)


@app.delete("/schedules/{schedule_id}", status_code=204)
async def delete_schedule(schedule_id: str):
    """Delete a schedule and cancel future runs."""
    try:
        await _scheduler.remove_schedule(schedule_id)
    except KeyError:
        raise HTTPException(404, detail=f"Schedule '{schedule_id}' not found")


@app.post("/schedules/{schedule_id}/pause", response_model=ScheduleResponse)
async def pause_schedule(schedule_id: str):
    """Pause a schedule (keeps it in the store but stops firing)."""
    try:
        record = await _scheduler.pause_schedule(schedule_id)
    except KeyError:
        raise HTTPException(404, detail=f"Schedule '{schedule_id}' not found")
    return _schedule_response(record)


@app.post("/schedules/{schedule_id}/resume", response_model=ScheduleResponse)
async def resume_schedule(schedule_id: str):
    """Resume a paused schedule."""
    try:
        record = await _scheduler.resume_schedule(schedule_id)
    except KeyError:
        raise HTTPException(404, detail=f"Schedule '{schedule_id}' not found")
    return _schedule_response(record)
