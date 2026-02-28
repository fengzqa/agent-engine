"""Tests for P7: Workflow Scheduler (cron / interval triggers)."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import api.server as server_module
from scheduler.models import ScheduleRecord, ScheduleStatus, ScheduleTriggerType
from scheduler.schedule_store import ScheduleStore
from scheduler.workflow_scheduler import WorkflowScheduler
from store.workflow_store import WorkflowStore
from tools.builtin.common import RunPythonTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner


# ── Shared workflow definition ────────────────────────────────────────────────

def simple_wf(name: str = "sched-test") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        steps=[
            StepDefinition(
                id="run",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "result = 'ok'"},
            )
        ],
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
async def stores(tmp_path):
    wf_store = WorkflowStore(f"sqlite+aiosqlite:///{tmp_path}/wf.db")
    await wf_store.init()
    sched_store = ScheduleStore(f"sqlite+aiosqlite:///{tmp_path}/sched.db")
    await sched_store.init()
    return wf_store, sched_store


@pytest.fixture
async def scheduler(stores):
    wf_store, sched_store = stores
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    runner = WorkflowRunner(reg, store=wf_store)
    sched = WorkflowScheduler(runner, sched_store, wf_store)
    await sched.start()
    yield sched
    await sched.shutdown()


@pytest.fixture(autouse=True)
async def isolated_server(tmp_path):
    """Patch the server module with per-test stores and a started scheduler."""
    wf_store = WorkflowStore(f"sqlite+aiosqlite:///{tmp_path}/wf.db")
    await wf_store.init()
    sched_store = ScheduleStore(f"sqlite+aiosqlite:///{tmp_path}/sched.db")
    await sched_store.init()

    runner = WorkflowRunner(server_module._tool_registry, store=wf_store)
    sched = WorkflowScheduler(runner, sched_store, wf_store)
    await sched.start()

    server_module._store = wf_store
    server_module._schedule_store = sched_store
    server_module._scheduler = sched
    yield
    await sched.shutdown()
    server_module._store = WorkflowStore()
    server_module._schedule_store = ScheduleStore()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=server_module.app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Unit tests: WorkflowScheduler ─────────────────────────────────────────────

async def test_add_schedule_persists(scheduler, stores):
    _, sched_store = stores
    record = ScheduleRecord(
        name="every-minute",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 1},
    )
    await scheduler.add_schedule(record)
    loaded = await sched_store.load(record.schedule_id)
    assert loaded.name == "every-minute"
    assert loaded.status == ScheduleStatus.ACTIVE


async def test_add_schedule_registers_job(scheduler):
    record = ScheduleRecord(
        name="test-job",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 5},
    )
    await scheduler.add_schedule(record)
    assert scheduler.next_run_time(record.schedule_id) is not None


async def test_fire_creates_workflow_instance(scheduler, stores):
    """Manually firing a schedule should create a workflow instance in the store."""
    wf_store, _ = stores
    record = ScheduleRecord(
        name="fire-test",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 1},
    )
    await scheduler.add_schedule(record)

    # Manually trigger the callback
    await scheduler._fire(record.schedule_id)
    await asyncio.sleep(0.1)  # let create_task run

    instances = await wf_store.list_all()
    assert len(instances) >= 1


async def test_interval_schedule_auto_fires(scheduler, stores):
    """A very short interval should fire at least once within a timeout."""
    wf_store, _ = stores
    record = ScheduleRecord(
        name="fast",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"seconds": 0.1},
    )
    await scheduler.add_schedule(record)
    await asyncio.sleep(0.5)  # let it fire at least once

    instances = await wf_store.list_all()
    assert len(instances) >= 1


async def test_pause_stops_next_run(scheduler):
    record = ScheduleRecord(
        name="pause-test",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 5},
    )
    await scheduler.add_schedule(record)
    await scheduler.pause_schedule(record.schedule_id)

    job = scheduler._aps.get_job(record.schedule_id)
    assert job is not None
    assert job.next_run_time is None  # paused jobs have no next_run_time


async def test_pause_persists_status(scheduler, stores):
    _, sched_store = stores
    record = ScheduleRecord(
        name="pause-persist",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 5},
    )
    await scheduler.add_schedule(record)
    await scheduler.pause_schedule(record.schedule_id)

    loaded = await sched_store.load(record.schedule_id)
    assert loaded.status == ScheduleStatus.PAUSED


async def test_resume_after_pause(scheduler):
    record = ScheduleRecord(
        name="resume-test",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 5},
    )
    await scheduler.add_schedule(record)
    await scheduler.pause_schedule(record.schedule_id)
    await scheduler.resume_schedule(record.schedule_id)

    assert scheduler.next_run_time(record.schedule_id) is not None


async def test_remove_schedule_deletes_job(scheduler, stores):
    _, sched_store = stores
    record = ScheduleRecord(
        name="delete-me",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 1},
    )
    await scheduler.add_schedule(record)
    await scheduler.remove_schedule(record.schedule_id)

    assert scheduler._aps.get_job(record.schedule_id) is None
    with pytest.raises(KeyError):
        await sched_store.load(record.schedule_id)


async def test_remove_nonexistent_raises(scheduler):
    with pytest.raises(KeyError):
        await scheduler.remove_schedule("does-not-exist")


async def test_scheduler_reloads_on_restart(stores):
    """Schedules saved to the store survive a scheduler restart."""
    wf_store, sched_store = stores
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    runner = WorkflowRunner(reg, store=wf_store)

    # First scheduler instance: add a schedule
    s1 = WorkflowScheduler(runner, sched_store, wf_store)
    await s1.start()
    record = ScheduleRecord(
        name="survive-restart",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.INTERVAL,
        trigger_config={"minutes": 1},
    )
    await s1.add_schedule(record)
    await s1.shutdown()

    # Second scheduler instance: should reload the schedule from store
    s2 = WorkflowScheduler(runner, sched_store, wf_store)
    await s2.start()
    assert s2.next_run_time(record.schedule_id) is not None
    await s2.shutdown()


async def test_cron_trigger_registers_job(scheduler):
    record = ScheduleRecord(
        name="daily",
        workflow_definition=simple_wf(),
        trigger_type=ScheduleTriggerType.CRON,
        trigger_config={"hour": 9, "minute": 0},
    )
    await scheduler.add_schedule(record)
    assert scheduler._aps.get_job(record.schedule_id) is not None


# ── API tests ─────────────────────────────────────────────────────────────────

async def test_api_create_schedule(client):
    resp = await client.post("/schedules", json={
        "name": "api-sched",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 10},
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "api-sched"
    assert data["trigger"] == "interval"
    assert "schedule_id" in data


async def test_api_list_schedules(client):
    await client.post("/schedules", json={
        "name": "s1",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 1},
    })
    await client.post("/schedules", json={
        "name": "s2",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 2},
    })
    resp = await client.get("/schedules")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_api_get_schedule(client):
    create = await client.post("/schedules", json={
        "name": "get-me",
        "definition": simple_wf().model_dump(),
        "trigger": "cron",
        "trigger_config": {"hour": 6, "minute": 0},
    })
    sid = create.json()["schedule_id"]
    resp = await client.get(f"/schedules/{sid}")
    assert resp.status_code == 200
    assert resp.json()["schedule_id"] == sid


async def test_api_get_schedule_not_found(client):
    resp = await client.get("/schedules/ghost")
    assert resp.status_code == 404


async def test_api_delete_schedule(client):
    create = await client.post("/schedules", json={
        "name": "del-me",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 1},
    })
    sid = create.json()["schedule_id"]
    resp = await client.delete(f"/schedules/{sid}")
    assert resp.status_code == 204
    assert (await client.get(f"/schedules/{sid}")).status_code == 404


async def test_api_delete_not_found(client):
    resp = await client.delete("/schedules/ghost")
    assert resp.status_code == 404


async def test_api_pause_schedule(client):
    create = await client.post("/schedules", json={
        "name": "pause-me",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 1},
    })
    sid = create.json()["schedule_id"]
    resp = await client.post(f"/schedules/{sid}/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"


async def test_api_resume_schedule(client):
    create = await client.post("/schedules", json={
        "name": "resume-me",
        "definition": simple_wf().model_dump(),
        "trigger": "interval",
        "trigger_config": {"minutes": 1},
    })
    sid = create.json()["schedule_id"]
    await client.post(f"/schedules/{sid}/pause")
    resp = await client.post(f"/schedules/{sid}/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


async def test_api_pause_not_found(client):
    resp = await client.post("/schedules/ghost/pause")
    assert resp.status_code == 404


async def test_api_schedule_in_openapi(client):
    resp = await client.get("/openapi.json")
    paths = set(resp.json()["paths"].keys())
    assert "/schedules" in paths
    assert "/schedules/{schedule_id}" in paths
    assert "/schedules/{schedule_id}/pause" in paths
    assert "/schedules/{schedule_id}/resume" in paths
