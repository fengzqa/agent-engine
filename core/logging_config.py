"""Structured JSON logging with trace_id propagation via contextvars."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

# Module-level ContextVar — propagates trace_id across async call trees
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)

# Fields that belong to LogRecord itself — we strip them from the "extra" dump
_STDLIB_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        data: dict = {
            "ts":       datetime.fromtimestamp(record.created, tz=timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level":    record.levelname,
            "logger":   record.name,
            "msg":      record.message,
            "trace_id": _trace_id_var.get(),
        }
        # Attach any extra= fields passed by the caller
        for key, val in record.__dict__.items():
            if key not in _STDLIB_FIELDS and not key.startswith("_"):
                data[key] = val

        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)

        return json.dumps(data, default=str)


def setup_json_logging(level: str = "INFO") -> None:
    """Replace root logger's handlers with a JSON formatter on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def set_trace_id(trace_id: str) -> None:
    """Bind a trace_id to the current async context."""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    return _trace_id_var.get()
