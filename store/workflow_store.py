"""WorkflowStore — SQLite-backed persistence for WorkflowInstance."""

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import create_async_engine

from core.state import WorkflowInstance

# ── Schema ───────────────────────────────────────────────────────────────────

_metadata = sa.MetaData()

_instances = sa.Table(
    "workflow_instances",
    _metadata,
    sa.Column("instance_id",   sa.String,  primary_key=True),
    sa.Column("workflow_name", sa.String,  nullable=False, index=True),
    sa.Column("status",        sa.String,  nullable=False),
    sa.Column("state_json",    sa.Text,    nullable=False),   # full Pydantic JSON
    sa.Column("updated_at",    sa.String,  nullable=False),
)


# ── Store ────────────────────────────────────────────────────────────────────

class WorkflowStore:
    """Persist and load WorkflowInstance objects via SQLite."""

    def __init__(self, db_url: str = "sqlite+aiosqlite:///workflows.db"):
        self._engine = create_async_engine(db_url, echo=False)

    async def init(self) -> None:
        """Create tables if they don't exist. Call once at startup."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def save(self, instance: WorkflowInstance) -> None:
        """Insert or update a workflow instance (upsert)."""
        from datetime import datetime, timezone
        row = {
            "instance_id":   instance.instance_id,
            "workflow_name": instance.workflow_name,
            "status":        instance.status.value,
            "state_json":    instance.model_dump_json(),
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }
        async with self._engine.begin() as conn:
            await conn.execute(
                sqlite_insert(_instances)
                .values(**row)
                .on_conflict_do_update(
                    index_elements=["instance_id"],
                    set_={k: row[k] for k in ("status", "state_json", "updated_at")},
                )
            )

    async def load(self, instance_id: str) -> WorkflowInstance:
        """Load a WorkflowInstance by ID. Raises KeyError if not found."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                sa.select(_instances).where(_instances.c.instance_id == instance_id)
            )).fetchone()
        if row is None:
            raise KeyError(f"Workflow instance '{instance_id}' not found")
        return WorkflowInstance.model_validate_json(row.state_json)

    async def list_all(self, workflow_name: str | None = None) -> list[dict]:
        """Return summary rows (no full state_json) ordered by most recent first."""
        cols = [
            _instances.c.instance_id,
            _instances.c.workflow_name,
            _instances.c.status,
            _instances.c.updated_at,
        ]
        query = sa.select(*cols)
        if workflow_name:
            query = query.where(_instances.c.workflow_name == workflow_name)
        query = query.order_by(_instances.c.updated_at.desc())

        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).fetchall()
        return [dict(r._mapping) for r in rows]

    async def delete(self, instance_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.delete(_instances).where(_instances.c.instance_id == instance_id)
            )
