"""WebSocket endpoint and static file serving for the live dashboard."""

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["dashboard"])
logger = structlog.get_logger()


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard events."""
    broadcaster = websocket.app.state.event_broadcaster
    await broadcaster.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
