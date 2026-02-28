"""Tool registry — register and look up tools by name."""

from tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Registered: {list(self._tools)}")
        return self._tools[name]

    def all_schemas(self) -> list[dict]:
        """Return all tool schemas for passing to Claude API."""
        return [t.to_api_schema() for t in self._tools.values()]
