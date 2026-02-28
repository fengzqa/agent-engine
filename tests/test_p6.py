"""Tests for P6: Multi-Agent collaboration (SUBWORKFLOW + MAP step types)."""

import pytest

from core.state import StepStatus, WorkflowStatus
from tools.builtin.common import RunPythonTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(RunPythonTool())
    return reg


@pytest.fixture
def runner(registry):
    return WorkflowRunner(registry)


# ── Helpers ───────────────────────────────────────────────────────────────────


def tool_step(step_id: str, code: str, **kwargs) -> StepDefinition:
    return StepDefinition(
        id=step_id,
        type=StepType.TOOL,
        tool="run_python",
        inputs={"code": code},
        **kwargs,
    )


# ── SUBWORKFLOW tests ─────────────────────────────────────────────────────────


async def test_subworkflow_step_completes(runner):
    """A SUBWORKFLOW step executes a nested workflow and returns step outputs."""
    sub_def = WorkflowDefinition(
        name="child",
        steps=[tool_step("greet", "result = 'hello'")],
    )
    parent = WorkflowDefinition(
        name="parent",
        steps=[
            StepDefinition(
                id="child_wf",
                type=StepType.SUBWORKFLOW,
                sub_workflow=sub_def,
            )
        ],
    )
    instance, _ = await runner.run(parent)
    assert instance.status == WorkflowStatus.COMPLETED
    child_out = instance.steps["child_wf"].output
    assert isinstance(child_out, dict)
    assert "greet" in child_out
    assert child_out["greet"] == "hello"


async def test_subworkflow_parallel_steps(runner):
    """Sub-workflow with parallel steps all complete before parent continues."""
    sub_def = WorkflowDefinition(
        name="parallel-child",
        steps=[
            tool_step("a", "result = 1"),
            tool_step("b", "result = 2"),
        ],
    )
    parent = WorkflowDefinition(
        name="parent",
        steps=[
            StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                sub_workflow=sub_def,
            )
        ],
    )
    instance, _ = await runner.run(parent)
    assert instance.status == WorkflowStatus.COMPLETED
    out = instance.steps["sub"].output
    assert out["a"] == 1
    assert out["b"] == 2


async def test_subworkflow_failure_propagates(runner):
    """If sub-workflow fails, the parent step is marked failed."""
    sub_def = WorkflowDefinition(
        name="bad-child",
        steps=[tool_step("boom", "raise RuntimeError('sub failed')")],
    )
    parent = WorkflowDefinition(
        name="parent",
        steps=[
            StepDefinition(
                id="sub",
                type=StepType.SUBWORKFLOW,
                sub_workflow=sub_def,
            )
        ],
    )
    instance, _ = await runner.run(parent)
    assert instance.status == WorkflowStatus.FAILED
    assert instance.steps["sub"].status == StepStatus.FAILED
    assert "sub failed" in instance.steps["sub"].error


async def test_subworkflow_output_used_by_downstream(runner):
    """Parent can reference sub-workflow output via {{child_wf.output}}."""
    sub_def = WorkflowDefinition(
        name="producer",
        steps=[tool_step("val", "result = 42")],
    )
    parent = WorkflowDefinition(
        name="consumer",
        steps=[
            StepDefinition(
                id="child_wf",
                type=StepType.SUBWORKFLOW,
                sub_workflow=sub_def,
            ),
            StepDefinition(
                id="use",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "data = {{child_wf.output}}\nresult = data['val'] * 2"},
                depends_on=["child_wf"],
            ),
        ],
    )
    instance, _ = await runner.run(parent)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["use"].output == 84


async def test_nested_subworkflow(runner):
    """Sub-workflow can itself contain another SUBWORKFLOW step."""
    inner = WorkflowDefinition(
        name="inner",
        steps=[tool_step("x", "result = 'deep'")],
    )
    middle = WorkflowDefinition(
        name="middle",
        steps=[
            StepDefinition(id="inner_wf", type=StepType.SUBWORKFLOW, sub_workflow=inner)
        ],
    )
    outer = WorkflowDefinition(
        name="outer",
        steps=[
            StepDefinition(id="mid_wf", type=StepType.SUBWORKFLOW, sub_workflow=middle)
        ],
    )
    instance, _ = await runner.run(outer)
    assert instance.status == WorkflowStatus.COMPLETED
    mid_out = instance.steps["mid_wf"].output
    assert mid_out["inner_wf"]["x"] == "deep"


# ── MAP tests ─────────────────────────────────────────────────────────────────


async def test_map_step_over_literal_list(runner):
    """MAP step with a hard-coded JSON list produces one output per item."""
    wf = WorkflowDefinition(
        name="map-test",
        steps=[
            StepDefinition(
                id="double",
                type=StepType.MAP,
                map_input='[1, 2, 3]',
                map_step=tool_step("compute", "result = {{item.output}} * 2"),
            )
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert sorted(instance.steps["double"].output) == [2, 4, 6]


async def test_map_step_empty_list(runner):
    """MAP over an empty list returns an empty list without errors."""
    wf = WorkflowDefinition(
        name="map-empty",
        steps=[
            StepDefinition(
                id="do_nothing",
                type=StepType.MAP,
                map_input='[]',
                map_step=tool_step("compute", "result = 1"),
            )
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["do_nothing"].output == []


async def test_map_input_from_previous_step(runner):
    """MAP can take its list from a prior step's output."""
    wf = WorkflowDefinition(
        name="map-from-step",
        steps=[
            tool_step("produce", "result = [10, 20, 30]"),
            StepDefinition(
                id="process",
                type=StepType.MAP,
                map_input="{{produce.output}}",
                map_step=tool_step("add_one", "result = {{item.output}} + 1"),
                depends_on=["produce"],
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert sorted(instance.steps["process"].output) == [11, 21, 31]


async def test_map_item_failure_propagates(runner):
    """If any item in a MAP step fails, the whole MAP step fails."""
    wf = WorkflowDefinition(
        name="map-fail",
        steps=[
            StepDefinition(
                id="boom",
                type=StepType.MAP,
                map_input='[1, 2]',
                map_step=tool_step("explode", "raise RuntimeError('item error')"),
            )
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.FAILED
    assert instance.steps["boom"].status == StepStatus.FAILED
    assert "item error" in instance.steps["boom"].error


async def test_map_downstream_receives_list(runner):
    """A step depending on a MAP step gets the list as its template input."""
    wf = WorkflowDefinition(
        name="map-downstream",
        steps=[
            StepDefinition(
                id="squares",
                type=StepType.MAP,
                map_input='[2, 3, 4]',
                map_step=tool_step("sq", "result = {{item.output}} ** 2"),
            ),
            StepDefinition(
                id="total",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "result = sum({{squares.output}})"},
                depends_on=["squares"],
            ),
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    assert instance.steps["total"].output == 29  # 4 + 9 + 16


async def test_map_with_subworkflow_map_step(runner):
    """MAP step where each item runs a SUBWORKFLOW."""
    sub_def = WorkflowDefinition(
        name="sub",
        steps=[tool_step("double", "result = 'processed'")],
    )
    wf = WorkflowDefinition(
        name="map-sub",
        steps=[
            StepDefinition(
                id="fan_out",
                type=StepType.MAP,
                map_input='["a", "b"]',
                map_step=StepDefinition(
                    id="run_sub",
                    type=StepType.SUBWORKFLOW,
                    sub_workflow=sub_def,
                ),
            )
        ],
    )
    instance, _ = await runner.run(wf)
    assert instance.status == WorkflowStatus.COMPLETED
    results = instance.steps["fan_out"].output
    assert len(results) == 2
    assert all(r["double"] == "processed" for r in results)


# ── DSL validation tests ──────────────────────────────────────────────────────


def test_subworkflow_step_requires_sub_workflow():
    with pytest.raises(ValueError, match="sub_workflow"):
        StepDefinition(id="s", type=StepType.SUBWORKFLOW)


def test_map_step_requires_map_input_and_map_step():
    with pytest.raises(ValueError, match="map_input"):
        StepDefinition(
            id="m",
            type=StepType.MAP,
            map_step=tool_step("x", "result = 1"),
        )
    with pytest.raises(ValueError, match="map_step"):
        StepDefinition(id="m", type=StepType.MAP, map_input='[1]')


def test_map_step_id_item_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        StepDefinition(
            id="m",
            type=StepType.MAP,
            map_input='[1]',
            map_step=tool_step("item", "result = 1"),
        )
