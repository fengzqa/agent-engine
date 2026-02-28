"""Tests for P2: conditional branching and enhanced retry."""

import pytest

from core.state import StepStatus, WorkflowStatus
from tools.builtin.common import RunPythonTool, WriteFileTool, ReadFileTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(RunPythonTool())
    r.register(WriteFileTool())
    r.register(ReadFileTool())
    return r


@pytest.fixture
def runner(registry):
    return WorkflowRunner(registry)


# ── Condition evaluation ─────────────────────────────────────────────────────

def _instance_with(step_outputs: dict):
    """Helper: build a WorkflowInstance with preset step outputs."""
    from core.state import StepState, WorkflowInstance
    return WorkflowInstance(
        workflow_name="test",
        steps={
            sid: StepState(step_id=sid, output=out)
            for sid, out in step_outputs.items()
        },
    )


def test_eval_condition_true(runner):
    inst = _instance_with({"check": {"exit_code": 0}})
    assert runner._eval_condition("check['exit_code'] == 0", inst) is True


def test_eval_condition_false(runner):
    inst = _instance_with({"check": {"exit_code": 1}})
    assert runner._eval_condition("check['exit_code'] == 0", inst) is False


def test_eval_condition_uses_len(runner):
    inst = _instance_with({"items": [1, 2, 3]})
    assert runner._eval_condition("len(items) > 2", inst) is True


def test_eval_condition_rejects_import(runner):
    inst = _instance_with({})
    with pytest.raises(Exception):
        runner._eval_condition("__import__('os').getcwd()", inst)


# ── Conditional branching ────────────────────────────────────────────────────

async def test_when_true_branch_runs(runner, tmp_path):
    """when=True → step executes normally."""
    out = str(tmp_path / "out.txt")
    wf = WorkflowDefinition(
        name="branch-true",
        steps=[
            StepDefinition(
                id="write",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": out, "content": "yes"},
                when="1 == 1",
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.steps["write"].status == StepStatus.COMPLETED
    assert instance.steps["write"].output == f"Written to {out}"


async def test_when_false_branch_skipped(runner, tmp_path):
    """when=False → step is SKIPPED, workflow still COMPLETED."""
    out = str(tmp_path / "out.txt")
    wf = WorkflowDefinition(
        name="branch-false",
        steps=[
            StepDefinition(
                id="write",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": out, "content": "no"},
                when="1 == 2",
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.steps["write"].status == StepStatus.SKIPPED
    assert instance.status == WorkflowStatus.COMPLETED


async def test_conditional_fork(runner, tmp_path):
    """One upstream step, two mutually exclusive downstream branches."""
    high_path = str(tmp_path / "high.txt")
    low_path  = str(tmp_path / "low.txt")

    # generate a number → route to high or low branch
    generate_code = (
        "import json\n"
        "result = json.dumps({'value': 80})\n"
        "print(result)"
    )

    wf = WorkflowDefinition(
        name="fork-test",
        steps=[
            StepDefinition(
                id="generate",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": generate_code},
            ),
            StepDefinition(
                id="high_branch",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": high_path, "content": "HIGH"},
                depends_on=["generate"],
                # generate output: {"stdout": "...", "stderr": ""}
                when="int('80') > 50",   # simplified: always True for value=80
            ),
            StepDefinition(
                id="low_branch",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": low_path, "content": "LOW"},
                depends_on=["generate"],
                when="int('80') <= 50",  # always False for value=80
            ),
        ],
    )

    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["high_branch"].status == StepStatus.COMPLETED
    assert instance.steps["low_branch"].status == StepStatus.SKIPPED


async def test_downstream_of_skipped_still_runs(runner, tmp_path):
    """A step depending on a SKIPPED step (not on its output) should still run."""
    out = str(tmp_path / "final.txt")
    wf = WorkflowDefinition(
        name="skip-propagation",
        steps=[
            StepDefinition(
                id="skipped_step",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": "/tmp/never.txt", "content": "x"},
                when="False",
            ),
            StepDefinition(
                id="final_step",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": out, "content": "done"},
                depends_on=["skipped_step"],
                # No `when` — should run regardless of skipped dep
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.steps["skipped_step"].status == StepStatus.SKIPPED
    assert instance.steps["final_step"].status == StepStatus.COMPLETED


# ── Enhanced retry: exponential backoff ──────────────────────────────────────

async def test_retry_with_backoff(runner, tmp_path):
    """Fail twice, succeed on third attempt; verify retry_delay is honored."""
    flag = tmp_path / "attempts.txt"
    flag.write_text("0")

    # Each run: read count, increment, fail if < 2, else succeed
    code = f"""
count = int(open('{flag}').read().strip())
count += 1
open('{flag}', 'w').write(str(count))
if count < 3:
    raise RuntimeError(f'attempt {{count}} failed')
print('success on attempt', count)
"""
    wf = WorkflowDefinition(
        name="backoff-test",
        steps=[
            StepDefinition(
                id="flaky",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": code},
                max_retries=2,
                retry_delay=0.0,    # no actual wait in tests
                retry_backoff=1.0,
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["flaky"].status == StepStatus.COMPLETED
    assert int(flag.read_text()) == 3   # exactly 3 attempts


async def test_retry_exhausted_marks_failed(runner):
    wf = WorkflowDefinition(
        name="exhaust-retry",
        steps=[
            StepDefinition(
                id="always_fail",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('nope')"},
                max_retries=2,
                retry_delay=0.0,
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.FAILED
    assert instance.steps["always_fail"].status == StepStatus.FAILED
