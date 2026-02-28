"""Tests for P5: FastAPI service layer."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import api.server as server_module
from store.workflow_store import WorkflowStore
from workflow.definition import StepDefinition, StepType, WorkflowDefinition


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def isolated_store(tmp_path):
    """Give each test its own in-memory store and reset the tool registry."""
    test_store = WorkflowStore(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await test_store.init()
    server_module._store = test_store
    yield
    # reset to avoid leaking state between test files
    server_module._store = WorkflowStore()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=server_module.app),
        base_url="http://test",
    ) as ac:
        yield ac


def fast_wf(name: str = "api-test") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        steps=[
            StepDefinition(
                id="step1",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('ok')"},
            ),
        ],
    )


async def wait_for_status(client, instance_id: str, target: str, timeout: float = 5.0):
    """Poll until workflow reaches target status or timeout.
    Raises AssertionError immediately if a different terminal state is reached.
    """
    _TERMINAL = {"completed", "failed"}
    deadline = asyncio.get_event_loop().time() + timeout
    last: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/workflows/{instance_id}")
        last = resp.json()
        status = last["status"]
        if status == target:
            return last
        if status in _TERMINAL:
            step_errors = {
                k: v.get("error") for k, v in last.get("steps", {}).items() if v.get("error")
            }
            raise AssertionError(
                f"Expected '{target}' but workflow reached '{status}'. "
                f"Step errors: {step_errors}"
            )
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Workflow did not reach '{target}' within {timeout}s. Last state: {last}")


# ── Health ────────────────────────────────────────────────────────────────────

async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── POST /workflows/run ───────────────────────────────────────────────────────

async def test_run_workflow_accepted(client):
    resp = await client.post(
        "/workflows/run",
        json={"definition": fast_wf().model_dump()},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "instance_id" in data
    assert data["status"] == "pending"


async def test_run_workflow_completes(client):
    resp = await client.post(
        "/workflows/run",
        json={"definition": fast_wf().model_dump()},
    )
    instance_id = resp.json()["instance_id"]
    result = await wait_for_status(client, instance_id, "completed")
    assert result["status"] == "completed"
    assert result["steps"]["step1"]["status"] == "completed"


async def test_run_invalid_workflow_rejected(client):
    """Workflow with unknown dep should fail schema validation before running."""
    bad_wf = {
        "name": "bad",
        "steps": [
            {
                "id": "a",
                "type": "tool",
                "tool": "run_python",
                "inputs": {"code": "pass"},
                "depends_on": ["nonexistent"],
            }
        ],
    }
    resp = await client.post("/workflows/run", json={"definition": bad_wf})
    assert resp.status_code == 422


# ── GET /workflows ────────────────────────────────────────────────────────────

async def test_list_workflows_empty(client):
    resp = await client.get("/workflows")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_workflows_returns_submitted(client):
    await client.post("/workflows/run", json={"definition": fast_wf("wf-a").model_dump()})
    await client.post("/workflows/run", json={"definition": fast_wf("wf-b").model_dump()})
    resp = await client.get("/workflows")
    assert len(resp.json()) == 2


async def test_list_workflows_filter_by_name(client):
    await client.post("/workflows/run", json={"definition": fast_wf("match").model_dump()})
    await client.post("/workflows/run", json={"definition": fast_wf("match").model_dump()})
    await client.post("/workflows/run", json={"definition": fast_wf("other").model_dump()})
    resp = await client.get("/workflows?workflow_name=match")
    assert len(resp.json()) == 2
    assert all(r["workflow_name"] == "match" for r in resp.json())


# ── GET /workflows/{instance_id} ─────────────────────────────────────────────

async def test_get_workflow_not_found(client):
    resp = await client.get("/workflows/nonexistent")
    assert resp.status_code == 404


async def test_get_workflow_returns_state(client):
    resp = await client.post(
        "/workflows/run",
        json={"definition": fast_wf().model_dump()},
    )
    instance_id = resp.json()["instance_id"]
    get_resp = await client.get(f"/workflows/{instance_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["instance_id"] == instance_id


# ── DELETE /workflows/{instance_id} ──────────────────────────────────────────

async def test_delete_workflow(client):
    resp = await client.post(
        "/workflows/run",
        json={"definition": fast_wf().model_dump()},
    )
    instance_id = resp.json()["instance_id"]

    del_resp = await client.delete(f"/workflows/{instance_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/workflows/{instance_id}")
    assert get_resp.status_code == 404


async def test_delete_not_found(client):
    resp = await client.delete("/workflows/ghost")
    assert resp.status_code == 404


# ── POST /workflows/{instance_id}/resume ─────────────────────────────────────

async def test_resume_not_found(client):
    resp = await client.post(
        "/workflows/ghost/resume",
        json={"definition": fast_wf().model_dump()},
    )
    assert resp.status_code == 404


async def test_resume_failed_workflow(client):
    """Submit a failing workflow, then resume with a fixed definition."""
    failing_wf = WorkflowDefinition(
        name="fail-resume",
        steps=[
            StepDefinition(
                id="bad",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('fail')"},
            ),
        ],
    )
    resp = await client.post(
        "/workflows/run",
        json={"definition": failing_wf.model_dump()},
    )
    instance_id = resp.json()["instance_id"]
    await wait_for_status(client, instance_id, "failed")

    fixed_wf = WorkflowDefinition(
        name="fail-resume",
        steps=[
            StepDefinition(
                id="bad",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('fixed')"},
            ),
        ],
    )
    resume_resp = await client.post(
        f"/workflows/{instance_id}/resume",
        json={"definition": fixed_wf.model_dump(), "retry_failed": True},
    )
    assert resume_resp.status_code == 202

    # Give the background task time to flip status from 'failed' → 'running'
    # before we start polling (one event-loop tick is enough).
    await asyncio.sleep(0.1)
    result = await wait_for_status(client, instance_id, "completed")
    assert result["status"] == "completed"


# ── OpenAPI schema ────────────────────────────────────────────────────────────

async def test_openapi_schema_available(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = set(schema["paths"].keys())
    assert "/workflows/run" in paths
    assert "/workflows/{instance_id}" in paths
    assert "/agent/chat" in paths
    assert "/health" in paths
