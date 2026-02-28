"""Tests for P9: CLI tool."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def wf_yaml(tmp_path):
    """Write a minimal workflow YAML file and return its path."""
    wf = {
        "name": "cli-test",
        "steps": [
            {
                "id": "s1",
                "type": "tool",
                "tool": "run_python",
                "inputs": {"code": "result = 'ok'"},
            }
        ],
    }
    p = tmp_path / "workflow.yaml"
    p.write_text(yaml.dump(wf))
    return str(p)


@pytest.fixture
def wf_json(tmp_path):
    """Write a minimal workflow JSON file and return its path."""
    wf = {
        "name": "cli-json-test",
        "steps": [
            {
                "id": "s1",
                "type": "tool",
                "tool": "run_python",
                "inputs": {"code": "result = 'ok'"},
            }
        ],
    }
    p = tmp_path / "workflow.json"
    p.write_text(json.dumps(wf))
    return str(p)


@pytest.fixture
def sched_yaml(tmp_path, wf_yaml):
    """Write a schedule YAML file and return its path."""
    sched = {
        "name": "test-schedule",
        "trigger": "interval",
        "trigger_config": {"minutes": 5},
        "definition": {
            "name": "sched-wf",
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "tool": "run_python",
                    "inputs": {"code": "result = 1"},
                }
            ],
        },
    }
    p = tmp_path / "schedule.yaml"
    p.write_text(yaml.dump(sched))
    return str(p)


def _mock_client(responses: dict):
    """
    Return a patched _client that maps (method, path_fragment) → httpx.Response.
    Accepts a dict of {method: response} or {path_fragment: response} for flexibility.
    """
    mock_c = MagicMock()
    mock_c.__enter__ = MagicMock(return_value=mock_c)
    mock_c.__exit__ = MagicMock(return_value=False)

    def _resp(status, data=None, text=""):
        r = MagicMock(spec=httpx.Response)
        r.status_code = status
        r.json = MagicMock(return_value=data or {})
        r.text = text
        r.is_error = status >= 400
        r.raise_for_status = MagicMock()
        return r

    mock_c._responses = responses
    mock_c._resp = _resp
    return mock_c


# ── engine --help ───────────────────────────────────��─────────────────────────


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "list" in result.output
    assert "status" in result.output
    assert "stream" in result.output
    assert "schedule" in result.output


def test_run_help(runner):
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--wait" in result.output


def test_schedule_help(runner):
    result = runner.invoke(cli, ["schedule", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "create" in result.output


# ── engine run ────────────────────────────────────────────────────────────────


def test_run_yaml(runner, wf_yaml):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"instance_id": "abc-123", "status": "pending"},
            is_error=False,
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["run", wf_yaml])
        assert result.exit_code == 0
        assert "abc-123" in result.output
        assert "pending" in result.output


def test_run_json_file(runner, wf_json):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"instance_id": "def-456", "status": "pending"},
            is_error=False,
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["run", wf_json])
        assert result.exit_code == 0
        assert "def-456" in result.output


def test_run_json_output(runner, wf_yaml):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"instance_id": "abc-123", "status": "pending"},
            is_error=False,
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["--json", "run", wf_yaml])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["instance_id"] == "abc-123"


def test_run_wait_completes(runner, wf_yaml):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"instance_id": "abc-123", "status": "pending"},
            is_error=False,
        )
        mc.get.return_value = MagicMock(
            status_code=200,
            is_error=False,
            json=lambda: {
                "instance_id": "abc-123",
                "workflow_name": "cli-test",
                "status": "completed",
                "duration_seconds": 1.23,
                "steps": {
                    "s1": {"status": "completed", "duration_seconds": 1.23, "error": None}
                },
            },
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["run", wf_yaml, "--wait"])
        assert result.exit_code == 0
        assert "completed" in result.output


def test_run_missing_file(runner):
    result = runner.invoke(cli, ["run", "nonexistent.yaml"])
    assert result.exit_code != 0


# ── engine list ───────────────────────────────────────────────────────────────


def test_list_empty(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: []
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No workflows" in result.output


def test_list_shows_instances(runner):
    rows = [
        {
            "instance_id": "abc-123",
            "workflow_name": "my-wf",
            "status": "completed",
            "duration_seconds": 1.5,
        }
    ]
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: rows
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "abc-123" in result.output
        assert "my-wf" in result.output


def test_list_json_output(runner):
    rows = [{"instance_id": "x", "workflow_name": "y", "status": "running"}]
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: rows
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["--json", "list"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["instance_id"] == "x"


# ── engine status ─────────────────────────────────────────────────────────────


def test_status_completed(runner):
    state = {
        "instance_id": "abc-123",
        "workflow_name": "my-wf",
        "status": "completed",
        "duration_seconds": 2.5,
        "steps": {
            "s1": {"status": "completed", "duration_seconds": 2.5, "error": None}
        },
    }
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: state
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["status", "abc-123"])
        assert result.exit_code == 0
        assert "completed" in result.output
        assert "s1" in result.output


def test_status_not_found(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=404,
            is_error=True,
            json=lambda: {"detail": "Instance 'xyz' not found"},
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["status", "xyz"])
        assert result.exit_code == 1


def test_status_json_output(runner):
    state = {
        "instance_id": "abc-123",
        "status": "failed",
        "steps": {},
        "duration_seconds": None,
    }
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: state
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["--json", "status", "abc-123"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "failed"


# ── engine schedule ───────────────────────────────────────────────────────────


def test_schedule_list_empty(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.get.return_value = MagicMock(
            status_code=200, is_error=False, json=lambda: []
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["schedule", "list"])
        assert result.exit_code == 0
        assert "No schedules" in result.output


def test_schedule_create(runner, sched_yaml):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=201,
            is_error=False,
            json=lambda: {
                "schedule_id": "sched-abc",
                "name": "test-schedule",
                "status": "active",
                "trigger": "interval",
                "trigger_config": {"minutes": 5},
                "next_run_at": None,
                "created_at": "2026-01-01T00:00:00",
            },
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["schedule", "create", sched_yaml])
        assert result.exit_code == 0
        assert "sched-abc" in result.output


def test_schedule_delete(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.delete.return_value = MagicMock(status_code=204, is_error=False)
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["schedule", "delete", "sched-abc"])
        assert result.exit_code == 0
        assert "Deleted" in result.output


def test_schedule_pause(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=200,
            is_error=False,
            json=lambda: {"schedule_id": "s1", "status": "paused"},
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["schedule", "pause", "s1"])
        assert result.exit_code == 0
        assert "Paused" in result.output


def test_schedule_resume(runner):
    with patch("cli.main._client") as mock_factory:
        mc = MagicMock()
        mc.__enter__ = MagicMock(return_value=mc)
        mc.__exit__ = MagicMock(return_value=False)
        mc.post.return_value = MagicMock(
            status_code=200,
            is_error=False,
            json=lambda: {"schedule_id": "s1", "status": "active"},
        )
        mock_factory.return_value = mc

        result = runner.invoke(cli, ["schedule", "resume", "s1"])
        assert result.exit_code == 0
        assert "Resumed" in result.output


# ── engine stream ─────────────────────────────────────────────────────────────


def test_stream_outputs_events(runner):
    """stream command parses SSE lines and prints status."""
    sse_lines = [
        'data: {"type":"state_change","instance_id":"abc","status":"running","steps":{"s1":{"status":"running"}},"timestamp":"t"}',
        'data: {"type":"state_change","instance_id":"abc","status":"completed","steps":{"s1":{"status":"completed"}},"timestamp":"t"}',
        "",
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(sse_lines))
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    mock_c = MagicMock()
    mock_c.stream = MagicMock(return_value=mock_resp)
    mock_c.__enter__ = MagicMock(return_value=mock_c)
    mock_c.__exit__ = MagicMock(return_value=False)

    with patch("cli.main.httpx.Client", return_value=mock_c):
        result = runner.invoke(cli, ["stream", "abc"])

    assert result.exit_code == 0
    assert "running" in result.output
    assert "completed" in result.output


def test_stream_json_output(runner):
    sse_lines = [
        'data: {"type":"state_change","instance_id":"abc","status":"completed","steps":{},"timestamp":"t"}',
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(sse_lines))
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    mock_c = MagicMock()
    mock_c.stream = MagicMock(return_value=mock_resp)
    mock_c.__enter__ = MagicMock(return_value=mock_c)
    mock_c.__exit__ = MagicMock(return_value=False)

    with patch("cli.main.httpx.Client", return_value=mock_c):
        result = runner.invoke(cli, ["--json", "stream", "abc"])

    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert data["status"] == "completed"
