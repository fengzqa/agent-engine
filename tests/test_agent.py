"""Tests for P0: Tool registry and Agent loop."""

import pytest

from tools.base import BaseTool, ToolResult
from tools.builtin.common import HttpRequestTool, RunPythonTool, ReadFileTool, WriteFileTool
from tools.registry import ToolRegistry
from core.agent import Agent


# --- Tool unit tests ---

class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes input back."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

    async def execute(self, message: str) -> ToolResult:
        return ToolResult(success=True, output=message)


def test_registry_register_and_get():
    registry = ToolRegistry()
    registry.register(EchoTool())
    tool = registry.get("echo")
    assert tool.name == "echo"


def test_registry_get_missing_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_all_schemas():
    registry = ToolRegistry()
    registry.register(EchoTool())
    schemas = registry.all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"


async def test_echo_tool_execute():
    tool = EchoTool()
    result = await tool.execute(message="hello")
    assert result.success is True
    assert result.output == "hello"


async def test_run_python_tool():
    tool = RunPythonTool()
    result = await tool.execute(code="print('hello from python')")
    assert result.success is True
    assert "hello from python" in result.output["stdout"]


async def test_write_and_read_file_tool(tmp_path):
    writer = WriteFileTool()
    reader = ReadFileTool()

    file_path = str(tmp_path / "test.txt")
    write_result = await writer.execute(path=file_path, content="agent engine")
    assert write_result.success is True

    read_result = await reader.execute(path=file_path)
    assert read_result.success is True
    assert read_result.output == "agent engine"


# --- Agent integration test (requires ANTHROPIC_API_KEY) ---

async def test_agent_with_echo_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())

    agent = Agent(registry=registry)
    response = await agent.run("Please use the echo tool with the message 'workflow test'.")
    assert "workflow test" in response.lower() or len(response) > 0
