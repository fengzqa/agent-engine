"""Tests for P3: WorkflowStore persistence and resume."""

import pytest

from core.state import StepStatus, WorkflowStatus
from store.workflow_store import WorkflowStore
from tools.builtin.common import RunPythonTool, WriteFileTool, ReadFileTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
async def store(tmp_path):
    s = WorkflowStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await s.init()
    return s


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(RunPythonTool())
    r.register(WriteFileTool())
    r.register(ReadFileTool())
    return r


@pytest.fixture
def runner(registry, store):
    return WorkflowRunner(registry, store=store)


def simple_wf(name: str = "test-wf") -> WorkflowDefinition:
    return WorkflowDefinition(
        name=name,
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('a')"},
            ),
            StepDefinition(
                id="step_b",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('b')"},
                depends_on=["step_a"],
            ),
        ],
    )


# ── WorkflowStore unit tests ─────────────────────────────────────────────────

async def test_store_save_and_load(store):
    from core.state import WorkflowInstance
    inst = WorkflowInstance(workflow_name="wf-save-test")
    await store.save(inst)
    loaded = await store.load(inst.instance_id)
    assert loaded.instance_id == inst.instance_id
    assert loaded.workflow_name == "wf-save-test"


async def test_store_upsert(store):
    from core.state import WorkflowInstance, WorkflowStatus
    inst = WorkflowInstance(workflow_name="wf-upsert")
    await store.save(inst)

    inst.status = WorkflowStatus.COMPLETED
    await store.save(inst)   # should update, not insert duplicate

    loaded = await store.load(inst.instance_id)
    assert loaded.status == WorkflowStatus.COMPLETED


async def test_store_load_missing_raises(store):
    with pytest.raises(KeyError):
        await store.load("nonexistent-id")


async def test_store_list_all(store):
    from core.state import WorkflowInstance
    inst1 = WorkflowInstance(workflow_name="wf-a")
    inst2 = WorkflowInstance(workflow_name="wf-b")
    await store.save(inst1)
    await store.save(inst2)

    rows = await store.list_all()
    assert len(rows) == 2
    names = {r["workflow_name"] for r in rows}
    assert names == {"wf-a", "wf-b"}


async def test_store_list_filter_by_name(store):
    from core.state import WorkflowInstance
    for _ in range(2):
        await store.save(WorkflowInstance(workflow_name="match"))
    await store.save(WorkflowInstance(workflow_name="other"))

    rows = await store.list_all(workflow_name="match")
    assert len(rows) == 2
    assert all(r["workflow_name"] == "match" for r in rows)


async def test_store_delete(store):
    from core.state import WorkflowInstance
    inst = WorkflowInstance(workflow_name="to-delete")
    await store.save(inst)
    await store.delete(inst.instance_id)
    with pytest.raises(KeyError):
        await store.load(inst.instance_id)


# ── Runner + Store integration ───────────────────────────────────────────────

async def test_run_persists_instance(runner, store):
    wf = simple_wf("persist-test")
    instance, _ = await runner.run(wf)

    loaded = await store.load(instance.instance_id)
    assert loaded.instance_id == instance.instance_id
    assert loaded.status == WorkflowStatus.COMPLETED


async def test_run_checkpoints_each_step(runner, store):
    """After run, all steps should be persisted as COMPLETED."""
    wf = simple_wf("checkpoint-test")
    instance, _ = await runner.run(wf)

    loaded = await store.load(instance.instance_id)
    for state in loaded.steps.values():
        assert state.status == StepStatus.COMPLETED


# ── Resume ───────────────────────────────────────────────────────────────────

async def test_resume_from_failed_step(runner, store, tmp_path):
    """Simulate a workflow that failed mid-run, then resume to completion."""
    out = str(tmp_path / "result.txt")

    wf = WorkflowDefinition(
        name="resume-test",
        steps=[
            StepDefinition(
                id="write",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": out, "content": "ok"},
            ),
            StepDefinition(
                id="fail_then_pass",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('simulated crash')"},
                depends_on=["write"],
            ),
        ],
    )

    # First run → fails at step 2
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.FAILED
    assert instance.steps["write"].status == StepStatus.COMPLETED
    assert instance.steps["fail_then_pass"].status == StepStatus.FAILED

    # Patch the failing step's code so resume succeeds
    wf.steps[1] = StepDefinition(
        id="fail_then_pass",
        type=StepType.TOOL,
        tool="run_python",
        inputs={"code": "print('recovered')"},
        depends_on=["write"],
    )

    # Resume — write step is already COMPLETED, only fail_then_pass reruns
    resumed, _ = await runner.resume(instance.instance_id, wf, retry_failed=True)
    assert resumed.status == WorkflowStatus.COMPLETED
    assert resumed.steps["write"].status == StepStatus.COMPLETED       # preserved
    assert resumed.steps["fail_then_pass"].status == StepStatus.COMPLETED  # rerun


async def test_resume_skip_completed_steps(runner, store, tmp_path):
    """COMPLETED steps are NOT re-executed on resume."""
    counter = tmp_path / "count.txt"
    counter.write_text("0")
    out = str(tmp_path / "out.txt")

    # Count how many times step_a runs
    count_code = f"""
n = int(open('{counter}').read()) + 1
open('{counter}', 'w').write(str(n))
open('{out}', 'w').write('done')
"""
    wf = WorkflowDefinition(
        name="skip-completed",
        steps=[
            StepDefinition(
                id="step_a",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": count_code},
            ),
            StepDefinition(
                id="step_b",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('fail')"},
                depends_on=["step_a"],
            ),
        ],
    )

    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.FAILED

    # Fix step_b and resume
    wf.steps[1] = StepDefinition(
        id="step_b",
        type=StepType.TOOL,
        tool="run_python",
        inputs={"code": "print('ok')"},
        depends_on=["step_a"],
    )
    _, __ = await runner.resume(instance.instance_id, wf, retry_failed=True)

    # step_a should have run exactly once (preserved from first run)
    assert int(counter.read_text()) == 1


async def test_resume_requires_store():
    """resume() without a store raises RuntimeError."""
    from tools.registry import ToolRegistry
    runner_no_store = WorkflowRunner(ToolRegistry())
    with pytest.raises(RuntimeError, match="store"):
        await runner_no_store.resume("any-id", simple_wf())
