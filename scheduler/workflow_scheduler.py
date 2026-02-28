"""APScheduler-backed workflow scheduler."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.state import WorkflowInstance, WorkflowStatus
from scheduler.models import ScheduleRecord, ScheduleStatus, ScheduleTriggerType
from scheduler.schedule_store import ScheduleStore

if TYPE_CHECKING:
    from store.workflow_store import WorkflowStore
    from workflow.runner import WorkflowRunner

logger = logging.getLogger(__name__)


class WorkflowScheduler:
    """Manages cron / interval triggers that launch workflow runs."""

    def __init__(
        self,
        runner: WorkflowRunner,
        schedule_store: ScheduleStore,
        workflow_store: WorkflowStore,
    ):
        self._runner = runner
        self._store = schedule_store
        self._wf_store = workflow_store
        self._aps = AsyncIOScheduler()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start APScheduler and reload all active schedules from the store."""
        self._aps.start()
        for record_dict in await self._store.list_all():
            record = ScheduleRecord.model_validate(record_dict)
            if record.status == ScheduleStatus.ACTIVE:
                self._register(record)
        logger.info("WorkflowScheduler started")

    async def shutdown(self) -> None:
        if self._aps.running:
            self._aps.shutdown(wait=False)
        logger.info("WorkflowScheduler stopped")

    # ── Schedule management ───────────────────────────────────────────────────

    async def add_schedule(self, record: ScheduleRecord) -> ScheduleRecord:
        """Persist a schedule and register it with APScheduler."""
        await self._store.save(record)
        if record.status == ScheduleStatus.ACTIVE:
            self._register(record)
        logger.info("Schedule added", extra={"schedule_id": record.schedule_id, "name": record.name})
        return record

    async def remove_schedule(self, schedule_id: str) -> None:
        """Delete a schedule from the store and deregister from APScheduler."""
        await self._store.load(schedule_id)   # raises KeyError if missing
        await self._store.delete(schedule_id)
        job = self._aps.get_job(schedule_id)
        if job:
            job.remove()
        logger.info("Schedule removed", extra={"schedule_id": schedule_id})

    async def pause_schedule(self, schedule_id: str) -> ScheduleRecord:
        record = await self._store.load(schedule_id)
        record.status = ScheduleStatus.PAUSED
        await self._store.save(record)
        job = self._aps.get_job(schedule_id)
        if job:
            job.pause()
        return record

    async def resume_schedule(self, schedule_id: str) -> ScheduleRecord:
        record = await self._store.load(schedule_id)
        record.status = ScheduleStatus.ACTIVE
        await self._store.save(record)
        job = self._aps.get_job(schedule_id)
        if job:
            job.resume()
        else:
            self._register(record)   # job missing → re-register
        return record

    def next_run_time(self, schedule_id: str) -> datetime | None:
        job = self._aps.get_job(schedule_id)
        return job.next_run_time if job else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _register(self, record: ScheduleRecord) -> None:
        if record.trigger_type == ScheduleTriggerType.CRON:
            trigger = CronTrigger(**record.trigger_config)
        else:
            trigger = IntervalTrigger(**record.trigger_config)

        self._aps.add_job(
            self._fire,
            trigger=trigger,
            id=record.schedule_id,
            kwargs={"schedule_id": record.schedule_id},
            replace_existing=True,
            max_instances=1,
        )

    async def _fire(self, schedule_id: str) -> None:
        """Callback invoked by APScheduler: create and run a workflow instance."""
        try:
            record = await self._store.load(schedule_id)
        except KeyError:
            logger.warning("Fired schedule not found in store", extra={"schedule_id": schedule_id})
            return

        instance = WorkflowInstance(
            workflow_name=record.workflow_definition.name,
            status=WorkflowStatus.PENDING,
            steps={},
        )
        await self._wf_store.save(instance)
        asyncio.create_task(
            self._runner.run(record.workflow_definition, instance_id=instance.instance_id)
        )

        record.last_run_at = datetime.now(timezone.utc)
        await self._store.save(record)
        logger.info(
            "Schedule fired",
            extra={"schedule_id": schedule_id, "instance_id": instance.instance_id},
        )
