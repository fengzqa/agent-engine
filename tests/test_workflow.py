"""Tests for P1: workflow definition, state machine, and runner."""

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


# ── WorkflowDefinition validation ───────────────────────────────────────────

def test_definition_rejects_unknown_dep():
    with pytest.raises(ValueError, match="unknown step"):
        WorkflowDefinition(
            name="bad",
            steps=[
                StepDefinition(id="a", type=StepType.LLM, prompt="hi", depends_on=["ghost"]),
            ],
        )


def test_definition_rejects_cycle():
    with pytest.raises(ValueError, match="cycle"):
        WorkflowDefinition(
            name="cyclic",
            steps=[
                StepDefinition(id="a", type=StepType.LLM, prompt="hi", depends_on=["b"]),
                StepDefinition(id="b", type=StepType.LLM, prompt="hi", depends_on=["a"]),
            ],
        )


def test_tool_step_requires_tool_field():
    with pytest.raises(ValueError):
        StepDefinition(id="x", type=StepType.TOOL)  # missing tool=


def test_llm_step_requires_prompt_field():
    with pytest.raises(ValueError):
        StepDefinition(id="x", type=StepType.LLM)  # missing prompt=


# ── Template resolution ──────────────────────────────────────────────────────

def test_template_resolution(runner):
    from core.state import StepState, WorkflowInstance
    instance = WorkflowInstance(
        workflow_name="test",
        steps={"step_a": StepState(step_id="step_a", output="hello world")},
    )
    result = runner._resolve("The output was: {{step_a.output}}", instance)
    assert result == "The output was: hello world"


def test_template_resolution_dict(runner):
    from core.state import StepState, WorkflowInstance
    instance = WorkflowInstance(
        workflow_name="test",
        steps={"s1": StepState(step_id="s1", output="42")},
    )
    result = runner._resolve({"value": "{{s1.output}}"}, instance)
    assert result == {"value": "42"}


# ── Runner: sequential workflow ──────────────────────────────────────────────

async def test_sequential_workflow(runner, tmp_path):
    """write_file → run_python → read result"""
    script = tmp_path / "add.py"
    output = tmp_path / "result.txt"

    wf = WorkflowDefinition(
        name="sequential-test",
        steps=[
            StepDefinition(
                id="write_script",
                type=StepType.TOOL,
                tool="write_file",
                inputs={"path": str(script), "content": f"open('{output}','w').write('99')"},
            ),
            StepDefinition(
                id="run_script",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": f"exec(open('{script}').read())"},
                depends_on=["write_script"],
            ),
            StepDefinition(
                id="read_result",
                type=StepType.TOOL,
                tool="read_file",
                inputs={"path": str(output)},
                depends_on=["run_script"],
            ),
        ],
    )

    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["write_script"].status == StepStatus.COMPLETED
    assert instance.steps["run_script"].status == StepStatus.COMPLETED
    assert instance.steps["read_result"].status == StepStatus.COMPLETED
    assert instance.steps["read_result"].output == "99"


# ── Runner: parallel steps ───────────────────────────────────────────────────

async def test_parallel_steps(runner, tmp_path):
    """Steps with no shared deps run in parallel."""
    f1 = str(tmp_path / "f1.txt")
    f2 = str(tmp_path / "f2.txt")

    wf = WorkflowDefinition(
        name="parallel-test",
        steps=[
            StepDefinition(id="w1", type=StepType.TOOL, tool="write_file",
                           inputs={"path": f1, "content": "A"}),
            StepDefinition(id="w2", type=StepType.TOOL, tool="write_file",
                           inputs={"path": f2, "content": "B"}),
            StepDefinition(id="read1", type=StepType.TOOL, tool="read_file",
                           inputs={"path": f1}, depends_on=["w1"]),
            StepDefinition(id="read2", type=StepType.TOOL, tool="read_file",
                           inputs={"path": f2}, depends_on=["w2"]),
        ],
    )

    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["read1"].output == "A"
    assert instance.steps["read2"].output == "B"


# ── Runner: step failure ─────────────────────────────────────────────────────

async def test_failed_step_marks_workflow_failed(runner):
    wf = WorkflowDefinition(
        name="fail-test",
        steps=[
            StepDefinition(
                id="bad",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "raise RuntimeError('intentional')"},
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.FAILED
    assert instance.steps["bad"].status == StepStatus.FAILED
    assert "intentional" in instance.steps["bad"].error


# ── Runner: retry ────────────────────────────────────────────────────────────

async def test_retry_succeeds_on_second_attempt(runner, tmp_path):
    """First run fails (file missing), second run succeeds after file is created."""
    flag = tmp_path / "flag.txt"

    # Code: fail if flag doesn't exist, create it so next attempt succeeds
    code = f"""
import os, sys
flag = '{flag}'
if not os.path.exists(flag):
    open(flag, 'w').close()
    raise RuntimeError('first attempt')
print('ok')
"""
    wf = WorkflowDefinition(
        name="retry-test",
        steps=[
            StepDefinition(
                id="flaky",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": code},
                max_retries=1,
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["flaky"].status == StepStatus.COMPLETED
