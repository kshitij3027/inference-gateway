"""Lightweight WebSocket event broadcaster for the live dashboard."""

import asyncio
import json
import time

import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class EventBroadcaster:
    """Broadcast events to all connected dashboard WebSocket clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("dashboard_ws_connected", clients=len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("dashboard_ws_disconnected", clients=len(self._connections))

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, event_type: str, data: dict) -> None:
        if not self._connections:
            return
        message = json.dumps({
            "type": event_type,
            "data": data,
            "ts": time.time(),
        })
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    def emit(self, event_type: str, data: dict) -> None:
        """Fire-and-forget broadcast via asyncio task. No-op if no clients."""
        if not self._connections:
            return
        asyncio.create_task(self.broadcast(event_type, data))
