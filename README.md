# Agent Engine

A self-built Python Agent Engine for workflow automation, powered by [Claude](https://www.anthropic.com/).

## Features

| Phase | Capability |
|-------|-----------|
| **P0** | Agent core loop — LLM + tool dispatch + result feed-back |
| **P1** | Workflow state machine — DAG execution, parallel steps, template variables |
| **P2** | Conditional branching (`when`) + exponential backoff retry |
| **P3** | Persistence & resume — SQLite checkpoint, recover from crash |
| **P4** | Observability — structured JSON logs, Span tracing, Token cost report |

## Project Structure

```
agent-engine/
├── core/
│   ├── agent.py          # Agent main loop (LLM ↔ tools)
│   ├── state.py          # WorkflowInstance / StepState types
│   ├── tracer.py         # Span collection + TokenUsage accounting
│   └── logging_config.py # JSON structured logging
├── tools/
│   ├── base.py           # BaseTool abstract class
│   ├── registry.py       # Tool registry
│   └── builtin/
│       └── common.py     # http_request, run_python, read_file, write_file
├── workflow/
│   ├── definition.py     # WorkflowDefinition DSL
│   └── runner.py         # WorkflowRunner execution engine
├── store/
│   └── workflow_store.py # SQLite persistence (SQLAlchemy async)
├── tests/                # 56 tests, all passing
├── main.py               # Interactive REPL entry point
└── workflow_demo.py      # End-to-end demo
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/fengzqa/agent-engine.git
cd agent-engine

# 2. Install dependencies
uv sync

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run interactive agent
uv run python main.py

# 5. Run workflow demo
uv run python workflow_demo.py
```

## Defining a Workflow

```python
from workflow.definition import WorkflowDefinition, StepDefinition, StepType

wf = WorkflowDefinition(
    name="data-pipeline",
    steps=[
        # TOOL step — call a registered tool directly
        StepDefinition(
            id="fetch",
            type=StepType.TOOL,
            tool="http_request",
            inputs={"method": "GET", "url": "https://api.example.com/data"},
        ),
        # LLM step — invoke Agent with a prompt
        StepDefinition(
            id="summarize",
            type=StepType.LLM,
            prompt="Summarize this data concisely:\n\n{{fetch.output}}",
            depends_on=["fetch"],
        ),
        # Conditional step — only runs when condition is True
        StepDefinition(
            id="save",
            type=StepType.TOOL,
            tool="write_file",
            inputs={"path": "/tmp/result.txt", "content": "{{summarize.output}}"},
            depends_on=["summarize"],
            when="len(summarize) > 0",
        ),
    ],
)
```

## Running a Workflow

```python
import asyncio
from tools.builtin.common import HttpRequestTool, RunPythonTool
from tools.registry import ToolRegistry
from workflow.runner import WorkflowRunner

async def main():
    registry = ToolRegistry()
    registry.register(HttpRequestTool())
    registry.register(RunPythonTool())

    runner = WorkflowRunner(registry)
    instance, tracer = await runner.run(wf)

    print(instance.summary())
    print(tracer.report(
        workflow_name=instance.workflow_name,
        instance_id=instance.instance_id,
    ).summary())

asyncio.run(main())
```

## Persistence & Resume

```python
from store.workflow_store import WorkflowStore

store = WorkflowStore("sqlite+aiosqlite:///workflows.db")
await store.init()

runner = WorkflowRunner(registry, store=store)

# Run — auto-checkpoints after every step
instance, tracer = await runner.run(wf)

# Resume after crash or failure
instance, tracer = await runner.resume(
    instance_id=instance.instance_id,
    definition=wf,
    retry_failed=True,
)
```

## Adding a Custom Tool

```python
from tools.base import BaseTool, ToolResult

class SendSlackTool(BaseTool):
    name = "send_slack"
    description = "Send a message to a Slack channel."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["channel", "message"],
        }

    async def execute(self, channel: str, message: str) -> ToolResult:
        # ... call Slack API
        return ToolResult(success=True, output="Message sent")

registry.register(SendSlackTool())
```

## Observability

Every workflow run returns a `Tracer` with full span and token data:

```
╔══ Trace Report ══════════════════════════════════════
  Trace ID  : 938b3c2c
  Workflow  : hello-pipeline
  Duration  : 2.09s

  Kind       Name                       Duration
  ────────────────────────────────────────────────
  step       fetch                         0.12s
  step       summarize                     2.05s
  llm        claude-sonnet-4-6             2.03s  (810 in / 95 out)

  Token Usage
  ────────────────────────────────────────────────
  Input tokens  :        810
  Output tokens :         95
  Total tokens  :        905
  Est. cost     : $    0.0039
╚══════════════════════════════════════════════════════
```

Logs are emitted as structured JSON with `trace_id` on every line:

```json
{"ts":"2026-02-28T10:00:01.234Z","level":"INFO","logger":"workflow.runner","msg":"Step completed","trace_id":"938b3c2c","step":"fetch","duration_s":0.12}
```

## Running Tests

```bash
uv run pytest tests/ -v
# 56 passed
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Anthropic API key
