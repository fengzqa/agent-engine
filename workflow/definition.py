"""Workflow DSL — define workflows as typed step graphs."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class StepType(str, Enum):
    TOOL = "tool"   # call a registered tool directly
    LLM  = "llm"   # invoke the Agent with a prompt (may use tools internally)


class StepDefinition(BaseModel):
    id: str
    type: StepType
    description: str = ""

    # TOOL step fields
    tool: str | None = None
    inputs: dict[str, Any] = {}

    # LLM step fields
    prompt: str | None = None

    # Execution control
    depends_on: list[str] = []
    max_retries: int = 0
    retry_delay: float = 0.0     # seconds before first retry
    retry_backoff: float = 1.0   # multiply delay by this factor each attempt

    # Conditional execution — Python expression evaluated against upstream outputs.
    # Context: each upstream step ID is bound to its output value.
    # Example: when="check['exit_code'] == 0"
    when: str | None = None

    @model_validator(mode="after")
    def check_required_fields(self):
        if self.type == StepType.TOOL and not self.tool:
            raise ValueError(f"Step '{self.id}': TOOL steps must set 'tool'")
        if self.type == StepType.LLM and not self.prompt:
            raise ValueError(f"Step '{self.id}': LLM steps must set 'prompt'")
        return self


class WorkflowDefinition(BaseModel):
    name: str
    description: str = ""
    steps: list[StepDefinition]

    @model_validator(mode="after")
    def validate_graph(self):
        ids = {s.id for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'"
                    )
        if self._has_cycle():
            raise ValueError("Workflow contains a dependency cycle")
        return self

    def _has_cycle(self) -> bool:
        """Kahn's algorithm for cycle detection."""
        dep_map = {s.id: set(s.depends_on) for s in self.steps}
        in_degree = {s.id: len(s.depends_on) for s in self.steps}
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for s in self.steps:
                if node in dep_map[s.id]:
                    dep_map[s.id].discard(node)
                    in_degree[s.id] -= 1
                    if in_degree[s.id] == 0:
                        queue.append(s.id)
        return visited != len(self.steps)
