"""Tool base class and type definitions."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: Any
    error: str | None = None


class BaseTool(ABC):
    """All tools must inherit from this class."""

    name: str
    description: str

    @abstractmethod
    def input_schema(self) -> dict:
        """Return JSON Schema for tool input parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool and return a result."""
        ...

    def to_api_schema(self) -> dict:
        """Convert to Anthropic tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema(),
        }
