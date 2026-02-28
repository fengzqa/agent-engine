"""Agent — core loop with tool-use support."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import anthropic

from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from core.tracer import Tracer

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10  # guard against infinite loops


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str = "You are a helpful workflow automation assistant.",
        model: str = MODEL,
        tracer: Tracer | None = None,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.registry = registry
        self.system_prompt = system_prompt
        self.model = model
        self.tracer = tracer
        self.messages: list[dict] = []

    async def run(self, user_input: str) -> str:
        """Run the agent loop until a final answer is produced."""
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(MAX_ITERATIONS):
            logger.debug("Iteration %d", iteration + 1)
            response = await self._call_llm()

            # Collect all text and tool_use blocks
            tool_use_blocks = []
            text_blocks = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_blocks.append(block)
                elif block.type == "text":
                    text_blocks.append(block.text)

            # Append assistant message (preserves all content blocks)
            self.messages.append({"role": "assistant", "content": response.content})

            # No tool calls → final answer
            if not tool_use_blocks:
                return "\n".join(text_blocks)

            # Execute all tool calls and collect results
            tool_results = []
            for block in tool_use_blocks:
                result = await self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
                logger.info("Tool '%s' → %s", block.name, result)

            self.messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached without a final answer."

    async def _call_llm(self) -> anthropic.types.Message:
        with (self.tracer.span(self.model, kind="llm") if self.tracer else _noop()) as span:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.registry.all_schemas(),
                messages=self.messages,
            )
            # Record token usage
            if response.usage and self.tracer:
                iu = response.usage.input_tokens
                ou = response.usage.output_tokens
                self.tracer.record_tokens(iu, ou)
                if span is not None:
                    span.attrs.update({"input_tokens": iu, "output_tokens": ou})
                logger.debug(
                    "LLM tokens",
                    extra={"input_tokens": iu, "output_tokens": ou, "model": self.model},
                )
            return response

    async def _execute_tool(self, name: str, inputs: dict) -> dict:
        try:
            tool = self.registry.get(name)
            result = await tool.execute(**inputs)
            return result.model_dump()
        except Exception as e:
            logger.error("Tool '%s' raised: %s", name, e)
            return {"success": False, "output": None, "error": str(e)}

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []


# ── Helpers ──────────────────────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def _noop():
    """Null context manager — used when tracer is None."""
    yield None
