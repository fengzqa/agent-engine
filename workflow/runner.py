"""Workflow runner — executes a WorkflowDefinition step by step."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.agent import Agent
from core.event_bus import EventBus
from core.logging_config import set_trace_id
from core.state import StepState, StepStatus, WorkflowInstance, WorkflowStatus
from core.tracer import Tracer
from tools.registry import ToolRegistry
from workflow.definition import StepDefinition, StepType, WorkflowDefinition

if TYPE_CHECKING:
    from store.workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

# Safe builtins available inside `when` expressions
_SAFE_BUILTINS = {
    "bool": bool, "int": int, "float": float, "str": str,
    "len": len, "any": any, "all": all, "min": min, "max": max, "sum": sum,
    "isinstance": isinstance, "hasattr": hasattr,
}

# Steps in these statuses unblock downstream steps
_DONE_STATUSES = {StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED}


class WorkflowRunner:
    def __init__(
        self,
        registry: ToolRegistry,
        store: WorkflowStore | None = None,
        tracer: Tracer | None = None,
        event_bus: EventBus | None = None,
    ):
        self.registry = registry
        self.store = store
        self.tracer = tracer
        self.event_bus = event_bus

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(
        self,
        definition: WorkflowDefinition,
        instance_id: str | None = None,
    ) -> tuple[WorkflowInstance, Tracer]:
        """Create a new workflow instance and execute it from scratch.

        Args:
            definition:  The workflow to execute.
            instance_id: Optional pre-assigned ID (used by the API layer so
                         callers can poll status before execution begins).

        Returns:
            (instance, tracer) — tracer holds spans and token usage for this run.
        """
        tracer = self.tracer or Tracer()
        set_trace_id(tracer.trace_id)

        init_kwargs: dict = dict(
            workflow_name=definition.name,
            status=WorkflowStatus.RUNNING,
            steps={s.id: StepState(step_id=s.id) for s in definition.steps},
        )
        if instance_id:
            init_kwargs["instance_id"] = instance_id
        instance = WorkflowInstance(**init_kwargs)
        logger.info(
            "Workflow started",
            extra={"workflow": definition.name, "instance_id": instance.instance_id},
        )
        await self._checkpoint(instance)

        with tracer.span(definition.name, kind="workflow",
                         instance_id=instance.instance_id):
            instance = await self._run_loop(instance, definition, tracer)

        return instance, tracer

    async def resume(
        self,
        instance_id: str,
        definition: WorkflowDefinition,
        retry_failed: bool = True,
    ) -> tuple[WorkflowInstance, Tracer]:
        """Load a persisted instance and continue from where it left off."""
        if not self.store:
            raise RuntimeError(
                "resume() requires a WorkflowStore — pass store= to WorkflowRunner"
            )

        tracer = self.tracer or Tracer()
        set_trace_id(tracer.trace_id)

        instance = await self.store.load(instance_id)
        logger.info(
            "Workflow resuming",
            extra={"workflow": instance.workflow_name, "instance_id": instance_id,
                   "retry_failed": retry_failed},
        )

        for state in instance.steps.values():
            if state.status == StepStatus.RUNNING:
                state.status = StepStatus.PENDING
                state.started_at = None
                state.finished_at = None
            elif retry_failed and state.status == StepStatus.FAILED:
                state.status = StepStatus.PENDING
                state.error = None
                state.started_at = None
                state.finished_at = None

        instance.status = WorkflowStatus.RUNNING
        instance.finished_at = None

        with tracer.span(definition.name, kind="workflow",
                         instance_id=instance.instance_id, resumed=True):
            instance = await self._run_loop(instance, definition, tracer)

        return instance, tracer

    # ── Core loop ────────────────────────────────────────────────────────────

    async def _run_loop(
        self,
        instance: WorkflowInstance,
        definition: WorkflowDefinition,
        tracer: Tracer,
    ) -> WorkflowInstance:
        step_map = {s.id: s for s in definition.steps}

        while True:
            ready = self._ready_steps(instance, step_map)
            if not ready:
                break

            await asyncio.gather(*[
                self._execute_step(instance, step_map[sid], tracer)
                for sid in ready
            ])

            if any(s.status == StepStatus.FAILED for s in instance.steps.values()):
                instance.status = WorkflowStatus.FAILED
                instance.finished_at = datetime.now(timezone.utc)
                await self._checkpoint(instance)
                logger.error("Workflow failed", extra={"workflow": definition.name})
                return instance

        instance.status = WorkflowStatus.COMPLETED
        instance.finished_at = datetime.now(timezone.utc)
        await self._checkpoint(instance)
        logger.info(
            "Workflow completed",
            extra={"workflow": definition.name, "duration_s": instance.duration_seconds},
        )
        return instance

    async def _checkpoint(self, instance: WorkflowInstance) -> None:
        if self.store:
            await self.store.save(instance)
        if self.event_bus:
            await self.event_bus.publish(instance.instance_id, _make_event(instance))

    # ── Step execution ───────────────────────────────────────────────────────

    def _ready_steps(self, instance: WorkflowInstance, step_map: dict) -> list[str]:
        return [
            sid for sid, state in instance.steps.items()
            if state.status == StepStatus.PENDING
            and all(
                instance.steps[dep].status in _DONE_STATUSES
                for dep in step_map[sid].depends_on
            )
        ]

    async def _execute_step(
        self,
        instance: WorkflowInstance,
        step: StepDefinition,
        tracer: Tracer,
    ) -> None:
        state = instance.steps[step.id]

        # ── Evaluate `when` condition ─────────────────────────────────────
        if step.when is not None:
            try:
                should_run = self._eval_condition(step.when, instance)
            except Exception as e:
                state.status = StepStatus.FAILED
                state.error = f"Condition error: {e}"
                state.finished_at = datetime.now(timezone.utc)
                logger.error("Step condition error", extra={"step": step.id, "error": str(e)})
                await self._checkpoint(instance)
                return

            if not should_run:
                state.status = StepStatus.SKIPPED
                logger.info("Step skipped", extra={"step": step.id, "when": step.when})
                await self._checkpoint(instance)
                return

        state.status = StepStatus.RUNNING
        state.started_at = datetime.now(timezone.utc)
        await self._checkpoint(instance)
        logger.info("Step started", extra={"step": step.id, "type": step.type.value})

        # ── Retry loop with exponential backoff ───────────────────────────
        delay = step.retry_delay
        for attempt in range(step.max_retries + 1):
            try:
                with tracer.span(step.id, kind="step",
                                 step_type=step.type.value, attempt=attempt):
                    state.output = await self._run_step(instance, step, tracer)

                state.status = StepStatus.COMPLETED
                state.finished_at = datetime.now(timezone.utc)
                await self._checkpoint(instance)
                logger.info(
                    "Step completed",
                    extra={"step": step.id, "duration_s": state.duration_seconds},
                )
                return
            except Exception as e:
                if attempt < step.max_retries:
                    logger.warning(
                        "Step retry",
                        extra={"step": step.id, "attempt": attempt + 1,
                               "max": step.max_retries + 1, "delay_s": delay},
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    delay *= step.retry_backoff
                else:
                    state.status = StepStatus.FAILED
                    state.error = str(e)
                    state.finished_at = datetime.now(timezone.utc)
                    await self._checkpoint(instance)
                    logger.error(
                        "Step failed",
                        extra={"step": step.id, "attempts": attempt + 1, "error": str(e)},
                    )

    async def _run_step(
        self,
        instance: WorkflowInstance,
        step: StepDefinition,
        tracer: Tracer,
    ) -> Any:
        if step.type == StepType.TOOL:
            resolved = self._resolve(step.inputs, instance)
            tool = self.registry.get(step.tool)
            result = await tool.execute(**resolved)
            if not result.success:
                raise RuntimeError(result.error)
            return result.output

        if step.type == StepType.LLM:
            resolved_prompt = self._resolve(step.prompt, instance)
            agent = Agent(registry=self.registry, tracer=tracer)
            return await agent.run(resolved_prompt)

        if step.type == StepType.SUBWORKFLOW:
            return await self._run_subworkflow(step.sub_workflow, tracer)

        if step.type == StepType.MAP:
            return await self._run_map(instance, step, tracer)

        raise ValueError(f"Unknown step type: {step.type}")

    async def _run_subworkflow(
        self,
        sub_def: WorkflowDefinition,
        tracer: Tracer,
    ) -> dict:
        """Execute a nested workflow in-memory and return its step outputs."""
        sub_runner = WorkflowRunner(self.registry, tracer=tracer)
        sub_instance, _ = await sub_runner.run(sub_def)
        if sub_instance.status.value == "failed":
            failed = {
                sid: s.error
                for sid, s in sub_instance.steps.items()
                if s.status.value == "failed"
            }
            raise RuntimeError(f"Sub-workflow '{sub_def.name}' failed: {failed}")
        return {sid: s.output for sid, s in sub_instance.steps.items()}

    async def _run_map(
        self,
        instance: WorkflowInstance,
        step: StepDefinition,
        tracer: Tracer,
    ) -> list:
        """Fan-out: run map_step for each item in the resolved list, in parallel."""
        raw = self._resolve(step.map_input, instance)
        if isinstance(raw, str):
            import json as _json
            items = _json.loads(raw)
        else:
            items = raw
        if not isinstance(items, list):
            raise ValueError(
                f"MAP step '{step.id}': map_input must resolve to a list, got {type(items).__name__}"
            )

        async def run_item(item: Any) -> Any:
            # Build a temporary WorkflowInstance that includes:
            # - parent step outputs (for cross-referencing with {{parent_step.output}})
            # - a synthetic "item" step whose output is the current element
            item_steps: dict[str, StepState] = {**instance.steps}
            item_steps["item"] = StepState(
                step_id="item",
                status=StepStatus.COMPLETED,
                output=item,
            )
            item_steps[step.map_step.id] = StepState(step_id=step.map_step.id)
            temp = WorkflowInstance(
                workflow_name=f"{instance.workflow_name}.__map__",
                status=WorkflowStatus.RUNNING,
                steps=item_steps,
            )
            await self._execute_step(temp, step.map_step, tracer)
            result_state = temp.steps[step.map_step.id]
            if result_state.status.value == "failed":
                raise RuntimeError(
                    f"MAP step '{step.id}' item failed: {result_state.error}"
                )
            return result_state.output

        return list(await asyncio.gather(*[run_item(item) for item in items]))

    # ── Expression evaluation ────────────────────────────────────────────────

    def _eval_condition(self, expr: str, instance: WorkflowInstance) -> bool:
        ctx = {sid: state.output for sid, state in instance.steps.items()}
        return bool(eval(expr, {"__builtins__": _SAFE_BUILTINS}, ctx))  # noqa: S307

    def _resolve(self, value: Any, instance: WorkflowInstance) -> Any:
        if isinstance(value, str):
            def replacer(m: re.Match) -> str:
                step_id = m.group(1)
                if step_id not in instance.steps:
                    raise ValueError(f"Unknown step reference '{{{{ {step_id}.output }}}}'")
                output = instance.steps[step_id].output
                return json.dumps(output) if not isinstance(output, str) else output
            return re.sub(r"\{\{(\w+)\.output\}\}", replacer, value)

        if isinstance(value, dict):
            return {k: self._resolve(v, instance) for k, v in value.items()}

        if isinstance(value, list):
            return [self._resolve(v, instance) for v in value]

        return value


# ── Module-level helpers ──────────────────────────────────────────────────────

def _make_event(instance: WorkflowInstance) -> dict:
    """Serialise a WorkflowInstance into an SSE-friendly dict."""
    return {
        "type": "state_change",
        "instance_id": instance.instance_id,
        "status": instance.status.value,
        "steps": {
            sid: {
                "status": s.status.value,
                "output": s.output,
                "error": s.error,
            }
            for sid, s in instance.steps.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
