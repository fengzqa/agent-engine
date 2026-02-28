"""Built-in tools: http_request, run_python, read_file, write_file."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import httpx

from tools.base import BaseTool, ToolResult


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = "Send an HTTP request and return the response body."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP method",
                },
                "url": {"type": "string", "description": "Full URL to request"},
                "headers": {
                    "type": "object",
                    "description": "Optional request headers",
                },
                "body": {
                    "type": "string",
                    "description": "Optional request body (JSON string)",
                },
            },
            "required": ["method", "url"],
        }

    async def execute(self, method: str, url: str, headers: dict | None = None, body: str | None = None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    content=body,
                )
                return ToolResult(
                    success=True,
                    output={"status_code": response.status_code, "body": response.text},
                )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class RunPythonTool(BaseTool):
    name = "run_python"
    description = "Execute a Python code snippet and return stdout/stderr."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        }

    async def execute(self, code: str) -> ToolResult:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ToolResult(
                success=result.returncode == 0,
                output={"stdout": result.stdout, "stderr": result.stderr},
                error=result.stderr if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Execution timed out (30s)")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read and return the contents of a file."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        try:
            content = Path(path).read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file, creating it if it doesn't exist."

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Written to {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
