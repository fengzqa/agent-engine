"""SQLite persistence for ScheduleRecord objects."""

import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from scheduler.models import ScheduleRecord


class ScheduleStore:
    def __init__(self, db_url: str = "sqlite+aiosqlite:///schedules.db"):
        self._engine = create_async_engine(db_url, echo=False)

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id TEXT PRIMARY KEY,
                    data        TEXT NOT NULL
                )
            """))

    async def save(self, record: ScheduleRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO schedules (schedule_id, data)
                    VALUES (:schedule_id, :data)
                    ON CONFLICT(schedule_id) DO UPDATE SET data=excluded.data
                """),
                {"schedule_id": record.schedule_id, "data": record.model_dump_json()},
            )

    async def load(self, schedule_id: str) -> ScheduleRecord:
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                text("SELECT data FROM schedules WHERE schedule_id = :id"),
                {"id": schedule_id},
            )).fetchone()
        if row is None:
            raise KeyError(schedule_id)
        return ScheduleRecord.model_validate_json(row[0])

    async def list_all(self) -> list[dict]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text("SELECT data FROM schedules ORDER BY rowid"))).fetchall()
        return [json.loads(r[0]) for r in rows]

    async def delete(self, schedule_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM schedules WHERE schedule_id = :id"),
                {"id": schedule_id},
            )

