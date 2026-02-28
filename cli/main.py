"""Agent Engine CLI — interact with a running Engine API server."""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import httpx
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

_TERMINAL = {"completed", "failed"}

_STATUS_COLOR: dict[str, str] = {
    "completed": "green",
    "failed": "red",
    "running": "yellow",
    "pending": "blue",
    "skipped": "dim",
    "active": "green",
    "paused": "yellow",
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _color(status: str) -> str:
    return _STATUS_COLOR.get(status, "white")


def _client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url.rstrip("/"), timeout=30)


def _load_file(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)


def _die(msg: str, code: int = 1) -> None:
    err_console.print(f"[red]Error:[/] {msg}")
    sys.exit(code)


def _check(resp: httpx.Response) -> None:
    if resp.status_code == 404:
        _die(resp.json().get("detail", "Not found"))
    if resp.is_error:
        _die(f"HTTP {resp.status_code}: {resp.text}")


# ── Root group ────────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--url", "-u",
    default="http://localhost:8000",
    envvar="ENGINE_URL",
    show_default=True,
    help="Engine API base URL.",
)
@click.option("--json", "json_output", is_flag=True, help="Output raw JSON.")
@click.pass_context
def cli(ctx: click.Context, url: str, json_output: bool) -> None:
    """Agent Engine — workflow automation CLI."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["json_output"] = json_output


# ── engine run ────────────────────────────────────────────────────────────────


@cli.command("run")
@click.argument("file", type=click.Path(exists=True))
@click.option("--wait", is_flag=True, help="Poll until the workflow finishes.")
@click.option("--interval", default=0.5, show_default=True, metavar="SECS",
              help="Poll interval when --wait is set.")
@click.pass_obj
def run(obj: dict, file: str, wait: bool, interval: float) -> None:
    """Submit a workflow definition (YAML or JSON file)."""
    definition = _load_file(file)
    with _client(obj["url"]) as c:
        resp = c.post("/workflows/run", json={"definition": definition})
    _check(resp)
    data = resp.json()

    if obj["json_output"]:
        click.echo(json.dumps(data, indent=2))
        return

    click.echo(f"Submitted  {data['instance_id']}  [{data['status']}]")

    if wait:
        import time
        iid = data["instance_id"]
        with _client(obj["url"]) as c:
            while True:
                r = c.get(f"/workflows/{iid}")
                _check(r)
                state = r.json()
                if state["status"] in _TERMINAL:
                    _print_status(state, obj["json_output"])
                    if state["status"] == "failed":
                        sys.exit(1)
                    return
                time.sleep(interval)


# ── engine list ───────────────────────────────────────────────────────────────


@cli.command("list")
@click.option("--name", help="Filter by workflow name.")
@click.pass_obj
def list_workflows(obj: dict, name: str | None) -> None:
    """List workflow instances."""
    params: dict[str, Any] = {}
    if name:
        params["workflow_name"] = name
    with _client(obj["url"]) as c:
        resp = c.get("/workflows", params=params)
    _check(resp)
    data = resp.json()

    if obj["json_output"]:
        click.echo(json.dumps(data, indent=2))
        return

    if not data:
        click.echo("No workflows found.")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Instance ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    for row in data:
        status = row.get("status", "?")
        dur = f"{row['duration_seconds']:.2f}s" if row.get("duration_seconds") else "-"
        table.add_row(
            row["instance_id"],
            row.get("workflow_name", ""),
            f"[{_color(status)}]{status}[/]",
            dur,
        )
    console.print(table)


# ── engine status ─────────────────────────────────────────────────────────────


@cli.command("status")
@click.argument("instance_id")
@click.pass_obj
def status(obj: dict, instance_id: str) -> None:
    """Show detailed status of a workflow instance."""
    with _client(obj["url"]) as c:
        resp = c.get(f"/workflows/{instance_id}")
    _check(resp)
    _print_status(resp.json(), obj["json_output"])


def _print_status(state: dict, json_output: bool) -> None:
    if json_output:
        click.echo(json.dumps(state, indent=2, default=str))
        return

    wf_status = state.get("status", "?")
    dur = f"  ({state['duration_seconds']:.2f}s)" if state.get("duration_seconds") else ""
    console.print(f"Status: [{_color(wf_status)}]{wf_status}[/]{dur}\n")

    steps = state.get("steps", {})
    if steps:
        table = Table(box=box.SIMPLE)
        table.add_column("Step", style="cyan")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        table.add_column("Error")
        for step_id, step in steps.items():
            s = step.get("status", "?")
            d = f"{step['duration_seconds']:.2f}s" if step.get("duration_seconds") else "-"
            err = (step.get("error") or "")[:60]
            table.add_row(
                step_id,
                f"[{_color(s)}]{s}[/]",
                d,
                f"[red]{err}[/]" if err else "",
            )
        console.print(table)


# ── engine stream ─────────────────────────────────────────────────────────────


@cli.command("stream")
@click.argument("instance_id")
@click.pass_obj
def stream(obj: dict, instance_id: str) -> None:
    """Stream live workflow state changes (SSE)."""
    url = obj["url"].rstrip("/") + f"/workflows/{instance_id}/stream"
    try:
        with httpx.Client(timeout=None) as c:
            with c.stream("GET", url) as resp:
                if resp.status_code == 404:
                    _die(f"Instance '{instance_id}' not found")
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if obj["json_output"]:
                        click.echo(json.dumps(event))
                    else:
                        wf_s = event.get("status", "?")
                        console.print(f"[{_color(wf_s)}][{wf_s}][/] workflow")
                        for sid, step in event.get("steps", {}).items():
                            s = step.get("status", "?")
                            console.print(f"  {sid}: [{_color(s)}]{s}[/]")
                    if event.get("status") in _TERMINAL:
                        break
    except httpx.ConnectError:
        _die(f"Cannot connect to {obj['url']}")


# ── engine schedule ──────────���────────────────────────────────────────────────


@cli.group("schedule")
def schedule() -> None:
    """Manage workflow schedules."""


@schedule.command("list")
@click.pass_obj
def schedule_list(obj: dict) -> None:
    """List all schedules."""
    with _client(obj["url"]) as c:
        resp = c.get("/schedules")
    _check(resp)
    data = resp.json()

    if obj["json_output"]:
        click.echo(json.dumps(data, indent=2))
        return

    if not data:
        click.echo("No schedules found.")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Schedule ID", style="cyan")
    table.add_column("Name")
    table.add_column("Trigger")
    table.add_column("Status")
    table.add_column("Next Run")
    for row in data:
        status = row.get("status", "?")
        table.add_row(
            row["schedule_id"],
            row.get("name", ""),
            row.get("trigger_type", row.get("trigger", "?")),
            f"[{_color(status)}]{status}[/]",
            row.get("next_run_at") or "-",
        )
    console.print(table)


@schedule.command("create")
@click.argument("file", type=click.Path(exists=True))
@click.pass_obj
def schedule_create(obj: dict, file: str) -> None:
    """Create a schedule from a YAML or JSON file.

    \b
    File format (YAML example):
      name: daily-job
      trigger: cron
      trigger_config:
        hour: 9
        minute: 0
      definition:
        name: my-workflow
        steps: [...]
    """
    payload = _load_file(file)
    with _client(obj["url"]) as c:
        resp = c.post("/schedules", json=payload)
    _check(resp)
    data = resp.json()

    if obj["json_output"]:
        click.echo(json.dumps(data, indent=2))
        return

    click.echo(f"Created  {data['schedule_id']}  [{data['status']}]  next: {data.get('next_run_at') or '-'}")


@schedule.command("delete")
@click.argument("schedule_id")
@click.pass_obj
def schedule_delete(obj: dict, schedule_id: str) -> None:
    """Delete a schedule."""
    with _client(obj["url"]) as c:
        resp = c.delete(f"/schedules/{schedule_id}")
    _check(resp)
    click.echo(f"Deleted  {schedule_id}")


@schedule.command("pause")
@click.argument("schedule_id")
@click.pass_obj
def schedule_pause(obj: dict, schedule_id: str) -> None:
    """Pause a schedule."""
    with _client(obj["url"]) as c:
        resp = c.post(f"/schedules/{schedule_id}/pause")
    _check(resp)
    click.echo(f"Paused  {schedule_id}")


@schedule.command("resume")
@click.argument("schedule_id")
@click.pass_obj
def schedule_resume(obj: dict, schedule_id: str) -> None:
    """Resume a paused schedule."""
    with _client(obj["url"]) as c:
        resp = c.post(f"/schedules/{schedule_id}/resume")
    _check(resp)
    click.echo(f"Resumed  {schedule_id}")
