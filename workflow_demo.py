"""Workflow demo — run a multi-step workflow from the command line."""

import asyncio
import logging
import sys

from tools.builtin.common import RunPythonTool, WriteFileTool, ReadFileTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(RunPythonTool())
    r.register(WriteFileTool())
    r.register(ReadFileTool())
    return r


# ── Example workflow: write a script, run it, read the output ────────────────

DEMO_WORKFLOW = WorkflowDefinition(
    name="hello-pipeline",
    description="Write a Python script, execute it, then read the result.",
    steps=[
        StepDefinition(
            id="write_script",
            type=StepType.TOOL,
            tool="write_file",
            inputs={
                "path": "/tmp/agent_demo.py",
                "content": (
                    "import json, math\n"
                    "result = {'pi': math.pi, 'sqrt2': math.sqrt(2)}\n"
                    "open('/tmp/agent_result.json', 'w').write(json.dumps(result, indent=2))\n"
                    "print('Script finished.')"
                ),
            },
        ),
        StepDefinition(
            id="run_script",
            type=StepType.TOOL,
            tool="run_python",
            inputs={"code": "exec(open('/tmp/agent_demo.py').read())"},
            depends_on=["write_script"],
        ),
        StepDefinition(
            id="read_result",
            type=StepType.TOOL,
            tool="read_file",
            inputs={"path": "/tmp/agent_result.json"},
            depends_on=["run_script"],
        ),
        StepDefinition(
            id="summarize",
            type=StepType.LLM,
            prompt=(
                "The workflow computed these math constants:\n\n"
                "{{read_result.output}}\n\n"
                "Write a one-sentence summary suitable for a report."
            ),
            depends_on=["read_result"],
        ),
    ],
)


async def main():
    registry = build_registry()
    runner = WorkflowRunner(registry)

    print(f"\nRunning workflow: {DEMO_WORKFLOW.name}\n{'─' * 50}")
    instance, tracer = await runner.run(DEMO_WORKFLOW)

    print(f"\n{'─' * 50}")
    print(instance.summary())

    print()
    print(tracer.report(
        workflow_name=instance.workflow_name,
        instance_id=instance.instance_id,
    ).summary())

    if instance.status.value == "completed":
        summary_output = instance.steps["summarize"].output
        print(f"\nLLM Summary:\n{summary_output}")
        sys.exit(0)
    else:
        for step_id, state in instance.steps.items():
            if state.error:
                print(f"\nFailed step '{step_id}': {state.error}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
