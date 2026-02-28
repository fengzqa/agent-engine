"""In-process async publish/subscribe event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict


class EventBus:
    """Lightweight pub/sub backed by asyncio.Queue.

    Keys are typically workflow instance IDs.  Multiple subscribers may
    register for the same key; each receives its own copy of every event.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, key: str) -> asyncio.Queue:
        """Return a queue that will receive all future events for *key*."""
        q: asyncio.Queue = asyncio.Queue()
        self._queues[key].append(q)
        return q

    def unsubscribe(self, key: str, q: asyncio.Queue) -> None:
        try:
            self._queues[key].remove(q)
        except ValueError:
            pass

    async def publish(self, key: str, event: dict) -> None:
        """Deliver *event* to every subscriber currently registered for *key*."""
        for q in list(self._queues.get(key, [])):
            await q.put(event)
