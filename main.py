"""Entry point — run the agent interactively or with a one-shot task."""

import asyncio
import logging
import os

from dotenv import load_dotenv

from core.agent import Agent
from tools.builtin.common import HttpRequestTool, ReadFileTool, RunPythonTool, WriteFileTool
from tools.registry import ToolRegistry

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(HttpRequestTool())
    registry.register(RunPythonTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    return registry


async def main():
    registry = build_registry()
    agent = Agent(
        registry=registry,
        system_prompt=(
            "You are a workflow automation assistant. "
            "Use the available tools to complete tasks accurately and efficiently. "
            "Always report what you did and the final result."
        ),
    )

    print("Agent Engine P0 — type 'exit' to quit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        print("Agent: thinking...", flush=True)
        response = await agent.run(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
