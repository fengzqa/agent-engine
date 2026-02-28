"""Tracer — lightweight span collection and token accounting."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

# ── Token pricing (claude-sonnet-4-6) ────────────────────────────────────────
_PRICE_INPUT_PER_TOKEN  = 3.00 / 1_000_000   # $3.00 / 1M input tokens
_PRICE_OUTPUT_PER_TOKEN = 15.00 / 1_000_000  # $15.00 / 1M output tokens


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Span:
    """A single timed operation."""
    name: str
    kind: str                        # workflow | step | llm | tool
    started_at: datetime
    finished_at: datetime | None = None
    attrs: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_s": self.duration_seconds,
            **self.attrs,
        }


@dataclass
class TokenUsage:
    """Accumulated token counts across all LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens  * _PRICE_INPUT_PER_TOKEN
            + self.output_tokens * _PRICE_OUTPUT_PER_TOKEN
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "cost_usd":      round(self.cost_usd(), 6),
        }


# ── TraceReport ───────────────────────────────────────────────────────────────

class TraceReport:
    def __init__(
        self,
        trace_id: str,
        workflow_name: str,
        instance_id: str,
        spans: list[Span],
        token_usage: TokenUsage,
    ):
        self.trace_id = trace_id
        self.workflow_name = workflow_name
        self.instance_id = instance_id
        self.spans = spans
        self.token_usage = token_usage

    @property
    def duration_seconds(self) -> float | None:
        wf_spans = [s for s in self.spans if s.kind == "workflow"]
        if wf_spans and wf_spans[0].duration_seconds is not None:
            return wf_spans[0].duration_seconds
        return None

    def summary(self) -> str:
        dur = f"{self.duration_seconds:.2f}s" if self.duration_seconds else "-"
        lines = [
            "╔══ Trace Report " + "═" * 38,
            f"  Trace ID  : {self.trace_id}",
            f"  Workflow  : {self.workflow_name}",
            f"  Instance  : {self.instance_id}",
            f"  Duration  : {dur}",
            "",
            f"  {'Kind':<10} {'Name':<24} {'Duration':>10}",
            "  " + "─" * 48,
        ]
        for span in self.spans:
            d = f"{span.duration_seconds:.2f}s" if span.duration_seconds else "-"
            extra = ""
            if span.kind == "llm":
                tin  = span.attrs.get("input_tokens", 0)
                tout = span.attrs.get("output_tokens", 0)
                extra = f"  ({tin:,} in / {tout:,} out)"
            lines.append(f"  {span.kind:<10} {span.name:<24} {d:>10}{extra}")

        lines += [
            "",
            "  Token Usage",
            "  " + "─" * 48,
            f"  Input tokens  : {self.token_usage.input_tokens:>10,}",
            f"  Output tokens : {self.token_usage.output_tokens:>10,}",
            f"  Total tokens  : {self.token_usage.total_tokens:>10,}",
            f"  Est. cost     : ${self.token_usage.cost_usd():>10.4f}  (claude-sonnet-4-6)",
            "╚" + "═" * 54,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "trace_id":      self.trace_id,
            "workflow_name": self.workflow_name,
            "instance_id":   self.instance_id,
            "duration_s":    self.duration_seconds,
            "spans":         [s.to_dict() for s in self.spans],
            "token_usage":   self.token_usage.to_dict(),
        }


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """Collects spans and token usage for one workflow run."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id: str = trace_id or uuid.uuid4().hex[:8]
        self.token_usage = TokenUsage()
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str, **attrs) -> Generator[Span, None, None]:
        """Context manager that records a timed span."""
        s = Span(name=name, kind=kind, started_at=datetime.now(timezone.utc), attrs=attrs)
        try:
            yield s
        finally:
            s.finished_at = datetime.now(timezone.utc)
            self._spans.append(s)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.token_usage.add(input_tokens, output_tokens)

    def report(self, workflow_name: str = "", instance_id: str = "") -> TraceReport:
        return TraceReport(
            trace_id=self.trace_id,
            workflow_name=workflow_name,
            instance_id=instance_id,
            spans=list(self._spans),
            token_usage=self.token_usage,
        )
