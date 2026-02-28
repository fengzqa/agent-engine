"""Tests for P4: Tracer, TokenUsage, JSON logging, and runner integration."""

import json
import logging

import pytest

from core.logging_config import JsonFormatter, set_trace_id
from core.tracer import Span, TokenUsage, Tracer
from tools.builtin.common import RunPythonTool, WriteFileTool
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition
from workflow.runner import WorkflowRunner


# ── TokenUsage ───────────────────────────────────────────────────────────────

def test_token_usage_accumulates():
    u = TokenUsage()
    u.add(100, 50)
    u.add(200, 100)
    assert u.input_tokens == 300
    assert u.output_tokens == 150
    assert u.total_tokens == 450


def test_token_usage_cost():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # $3/M input + $15/M output = $18
    assert abs(u.cost_usd() - 18.0) < 1e-6


def test_token_usage_to_dict():
    u = TokenUsage(input_tokens=10, output_tokens=5)
    d = u.to_dict()
    assert d["total_tokens"] == 15
    assert "cost_usd" in d


# ── Span ─────────────────────────────────────────────────────────────────────

def test_span_duration():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    s = Span(name="x", kind="step", started_at=now,
             finished_at=now + timedelta(seconds=2.5))
    assert abs(s.duration_seconds - 2.5) < 0.01


def test_span_to_dict():
    from datetime import datetime, timezone
    s = Span(name="my-step", kind="step",
             started_at=datetime.now(timezone.utc), attrs={"foo": "bar"})
    s.finished_at = datetime.now(timezone.utc)
    d = s.to_dict()
    assert d["name"] == "my-step"
    assert d["kind"] == "step"
    assert d["foo"] == "bar"
    assert "duration_s" in d


# ── Tracer ───────────────────────────────────────────────────────────────────

def test_tracer_span_context_manager():
    t = Tracer()
    with t.span("step-a", kind="step"):
        pass
    assert len(t._spans) == 1
    assert t._spans[0].name == "step-a"
    assert t._spans[0].finished_at is not None


def test_tracer_span_attrs():
    t = Tracer()
    with t.span("llm-call", kind="llm", model="sonnet") as s:
        s.attrs["tokens"] = 42
    assert t._spans[0].attrs["tokens"] == 42
    assert t._spans[0].attrs["model"] == "sonnet"


def test_tracer_record_tokens():
    t = Tracer()
    t.record_tokens(100, 50)
    t.record_tokens(200, 100)
    assert t.token_usage.input_tokens == 300
    assert t.token_usage.output_tokens == 150


def test_tracer_trace_id_unique():
    assert Tracer().trace_id != Tracer().trace_id


def test_tracer_custom_trace_id():
    t = Tracer(trace_id="abc123")
    assert t.trace_id == "abc123"


def test_tracer_report_contains_spans():
    t = Tracer()
    with t.span("wf", kind="workflow"):
        with t.span("s1", kind="step"):
            pass
    t.record_tokens(500, 200)
    report = t.report(workflow_name="test-wf", instance_id="inst-1")
    assert report.workflow_name == "test-wf"
    assert len(report.spans) == 2
    assert report.token_usage.input_tokens == 500


def test_trace_report_summary_contains_key_fields():
    t = Tracer(trace_id="deadbeef")
    with t.span("my-workflow", kind="workflow"):
        pass
    t.record_tokens(1024, 512)
    summary = t.report(workflow_name="my-workflow", instance_id="abc").summary()
    assert "deadbeef" in summary
    assert "my-workflow" in summary
    assert "1,024" in summary   # input tokens formatted
    assert "512" in summary     # output tokens


def test_trace_report_to_dict():
    t = Tracer()
    with t.span("wf", kind="workflow"):
        pass
    d = t.report().to_dict()
    assert "spans" in d
    assert "token_usage" in d
    assert isinstance(d["spans"], list)


# ── JSON logging ─────────────────────────────────────────────────────────────

def _make_record(msg: str, level=logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_valid_json():
    fmt = JsonFormatter()
    output = fmt.format(_make_record("hello world"))
    parsed = json.loads(output)
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed
    assert "logger" in parsed


def test_json_formatter_includes_trace_id():
    set_trace_id("trace-xyz")
    fmt = JsonFormatter()
    parsed = json.loads(fmt.format(_make_record("hi")))
    assert parsed["trace_id"] == "trace-xyz"


def test_json_formatter_extra_fields():
    fmt = JsonFormatter()
    parsed = json.loads(fmt.format(_make_record("step done", step="write", duration_s=0.05)))
    assert parsed["step"] == "write"
    assert parsed["duration_s"] == 0.05


# ── Runner integration: tracer wired end-to-end ───────────────────────────────

@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(RunPythonTool())
    r.register(WriteFileTool())
    return r


async def test_runner_returns_tracer(registry):
    runner = WorkflowRunner(registry)
    wf = WorkflowDefinition(
        name="tracer-test",
        steps=[
            StepDefinition(
                id="step1",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('ok')"},
            ),
        ],
    )
    instance, tracer = await runner.run(wf)
    assert isinstance(tracer, Tracer)
    # Should have a workflow span + at least one step span
    kinds = {s.kind for s in tracer._spans}
    assert "workflow" in kinds
    assert "step" in kinds


async def test_runner_tracer_spans_have_durations(registry):
    runner = WorkflowRunner(registry)
    wf = WorkflowDefinition(
        name="duration-test",
        steps=[
            StepDefinition(
                id="s1",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('a')"},
            ),
            StepDefinition(
                id="s2",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "print('b')"},
                depends_on=["s1"],
            ),
        ],
    )
    _, tracer = await runner.run(wf)
    for span in tracer._spans:
        assert span.duration_seconds is not None
        assert span.duration_seconds >= 0


async def test_runner_injected_tracer_is_reused(registry):
    """When caller injects a Tracer, runner uses it (not a new one)."""
    my_tracer = Tracer(trace_id="injected")
    runner = WorkflowRunner(registry, tracer=my_tracer)
    wf = WorkflowDefinition(
        name="injected-tracer",
        steps=[
            StepDefinition(
                id="s",
                type=StepType.TOOL,
                tool="run_python",
                inputs={"code": "pass"},
            ),
        ],
    )
    _, returned_tracer = await runner.run(wf)
    assert returned_tracer is my_tracer
    assert returned_tracer.trace_id == "injected"
