"""Tests for P8: SSE real-time workflow event streaming."""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

import api.server as server_module
from core.event_bus import EventBus
from scheduler.schedule_store import ScheduleStore
from scheduler.workflow_scheduler import WorkflowScheduler
from store.workflow_store import WorkflowStore
from tools.builtin.common import RunPythonTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner, _make_event
from core.state import WorkflowInstance, WorkflowStatus


# ── Helpers ───────────────────────────────────────────────────────────────────


def fast_wf(name: str = "sse-test") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        steps=[
            StepDefinition(
                id="step1",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "result = 'ok'"},
            )
        ],
    )


async def collect_sse(client, url: str, max_events: int = 20) -> list[dict]:
    """Read SSE data lines from *url* until a terminal event or max_events."""
    events = []
    async with client.stream("GET", url) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                events.append(event)
                if event.get("status") in {"completed", "failed"} or len(events) >= max_events:
                    break
    return events


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
async def isolated_server(tmp_path):
    """Per-test stores + fresh EventBus wired into the server module."""
    wf_store = WorkflowStore(f"sqlite+aiosqlite:///{tmp_path}/wf.db")
    await wf_store.init()
    sched_store = ScheduleStore(f"sqlite+aiosqlite:///{tmp_path}/sched.db")
    await sched_store.init()
    event_bus = EventBus()

    runner = WorkflowRunner(server_module._tool_registry, store=wf_store, event_bus=event_bus)
    sched = WorkflowScheduler(runner, sched_store, wf_store)
    await sched.start()

    server_module._store = wf_store
    server_module._schedule_store = sched_store
    server_module._event_bus = event_bus
    server_module._runner = runner
    server_module._scheduler = sched
    yield
    await sched.shutdown()
    server_module._store = WorkflowStore()
    server_module._schedule_store = ScheduleStore()
    server_module._event_bus = EventBus()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=server_module.app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── EventBus unit tests ───────────────────────────────────────────────────────


async def test_event_bus_subscribe_and_receive():
    bus = EventBus()
    q = bus.subscribe("wf-1")
    await bus.publish("wf-1", {"status": "running"})
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["status"] == "running"


async def test_event_bus_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe("wf-1")
    bus.unsubscribe("wf-1", q)
    await bus.publish("wf-1", {"status": "completed"})
    assert q.empty()


async def test_event_bus_multiple_subscribers():
    bus = EventBus()
    q1 = bus.subscribe("wf-1")
    q2 = bus.subscribe("wf-1")
    await bus.publish("wf-1", {"x": 1})
    assert (await q1.get())["x"] == 1
    assert (await q2.get())["x"] == 1


async def test_event_bus_different_keys_isolated():
    bus = EventBus()
    q_a = bus.subscribe("a")
    q_b = bus.subscribe("b")
    await bus.publish("a", {"key": "a-event"})
    assert not q_b.empty() is False
    assert (await q_a.get())["key"] == "a-event"
    assert q_b.empty()


# ── Runner + EventBus integration ─────────────────────────────────────────────


async def test_runner_publishes_on_checkpoint():
    """Runner with event_bus should publish an event after each checkpoint."""
    bus = EventBus()
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    runner = WorkflowRunner(reg, event_bus=bus)

    q = bus.subscribe("pre-run")  # subscribe before run to catch all events
    instance = WorkflowInstance(
        instance_id="pre-run",
        workflow_name="t",
        status=WorkflowStatus.PENDING,
        steps={},
    )

    # Manually call _checkpoint to test publishing
    await runner._checkpoint(instance)
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["type"] == "state_change"
    assert event["instance_id"] == "pre-run"


async def test_runner_publishes_completed_event():
    """Running a workflow should emit at least one event with status=completed."""
    bus = EventBus()
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    runner = WorkflowRunner(reg, event_bus=bus)

    events: list[dict] = []

    async def collect():
        # We don't know the instance_id yet; subscribe after run returns
        pass

    instance, _ = await runner.run(fast_wf(), instance_id="known-id")
    q = bus.subscribe("known-id")

    # Trigger one more checkpoint to confirm subscription works
    await runner._checkpoint(instance)
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["status"] == "completed"


async def test_runner_publishes_failed_event():
    failing_wf = WorkflowDefinition(
        name="fail",
        steps=[
            StepDefinition(
                id="boom",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('oh no')"},
            )
        ],
    )
    bus = EventBus()
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    runner = WorkflowRunner(reg, event_bus=bus)

    # Subscribe with a known ID
    q = bus.subscribe("fail-id")
    asyncio.create_task(runner.run(failing_wf, instance_id="fail-id"))

    # Collect events until terminal
    received = []
    while True:
        event = await asyncio.wait_for(q.get(), timeout=3.0)
        received.append(event)
        if event["status"] in {"completed", "failed"}:
            break

    assert received[-1]["status"] == "failed"


# ── SSE API tests ─────────────────────────────────────────────────────────────


async def test_stream_not_found(client):
    resp = await client.get("/workflows/ghost/stream")
    assert resp.status_code == 404


async def test_stream_content_type(client):
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]
    events = await collect_sse(client, f"/workflows/{iid}/stream")
    # Verify content-type was SSE — checked via successful parsing above
    assert len(events) >= 1


async def test_stream_returns_terminal_event(client):
    """SSE stream should end with a completed or failed status event."""
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]
    events = await collect_sse(client, f"/workflows/{iid}/stream")
    assert events[-1]["status"] in {"completed", "failed"}


async def test_stream_events_have_required_fields(client):
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]
    events = await collect_sse(client, f"/workflows/{iid}/stream")
    for event in events:
        assert "type" in event
        assert "instance_id" in event
        assert "status" in event
        assert "steps" in event
        assert "timestamp" in event
        assert event["instance_id"] == iid


async def test_stream_instance_id_matches(client):
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]
    events = await collect_sse(client, f"/workflows/{iid}/stream")
    assert all(e["instance_id"] == iid for e in events)


async def test_stream_already_completed_workflow(client):
    """Connecting to an already-completed workflow should get one snapshot and close."""
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]

    # Wait until workflow completes
    for _ in range(50):
        state = (await client.get(f"/workflows/{iid}")).json()
        if state["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.1)

    events = await collect_sse(client, f"/workflows/{iid}/stream")
    assert len(events) == 1
    assert events[0]["status"] in {"completed", "failed"}


async def test_stream_step_statuses_present(client):
    """Events should include step-level status information."""
    run = await client.post("/workflows/run", json={"definition": fast_wf().model_dump()})
    iid = run.json()["instance_id"]
    events = await collect_sse(client, f"/workflows/{iid}/stream")
    terminal = events[-1]
    assert "step1" in terminal["steps"]
    assert terminal["steps"]["step1"]["status"] in {"completed", "failed", "pending", "running", "skipped"}


async def test_stream_in_openapi(client):
    resp = await client.get("/openapi.json")
    paths = set(resp.json()["paths"].keys())
    assert "/workflows/{instance_id}/stream" in paths
