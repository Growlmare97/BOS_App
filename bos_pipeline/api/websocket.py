"""WebSocket endpoint for real-time pipeline progress updates.

Clients connect to ws://localhost:8000/ws/{job_id} and receive JSON messages:
  {"job_id": "...", "stage": "displacement", "progress": 45, "message": "Frame 3/10"}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Tracks live WebSocket connections keyed by job_id."""

    def __init__(self) -> None:
        # job_id -> list of connected sockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        """Accept *websocket* and register it under *job_id*."""
        await websocket.accept()
        self.active_connections.setdefault(job_id, []).append(websocket)
        logger.debug("WS connected: job_id=%s  total=%d", job_id,
                     len(self.active_connections[job_id]))

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        """Remove *websocket* from the registry; clean up empty job buckets."""
        conns = self.active_connections.get(job_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self.active_connections.pop(job_id, None)
        logger.debug("WS disconnected: job_id=%s", job_id)

    async def broadcast(self, job_id: str, message: dict) -> None:
        """Send *message* as JSON to every WebSocket registered for *job_id*.

        Dead connections are silently removed during the broadcast.
        """
        conns = self.active_connections.get(job_id, [])
        if not conns:
            return

        dead: List[WebSocket] = []
        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception as exc:
                logger.debug("WS send failed (%s); marking dead.", exc)
                dead.append(ws)

        for ws in dead:
            self.disconnect(job_id, ws)


# Module-level singleton shared across the application.
manager = ConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str) -> None:
    """Accept a WebSocket connection for *job_id* and keep it alive."""
    await manager.connect(job_id, websocket)
    try:
        while True:
            # Keep the connection open; the server pushes messages via broadcast.
            # We still need to receive to detect client-side disconnects.
            await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        # Heartbeat timeout — send a ping so the client knows we're alive.
        try:
            await websocket.send_json({"type": "ping"})
        except Exception:
            pass
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected gracefully: job_id=%s", job_id)
    except Exception as exc:
        logger.warning("WebSocket error (job_id=%s): %s", job_id, exc)
    finally:
        manager.disconnect(job_id, websocket)


# ---------------------------------------------------------------------------
# Convenience coroutine for use from background tasks
# ---------------------------------------------------------------------------


async def broadcast_progress(
    job_id: str,
    stage: str,
    progress: int,
    message: str,
) -> None:
    """Broadcast a progress update to all WebSocket clients watching *job_id*.

    Parameters
    ----------
    job_id:
        Identifier of the running job.
    stage:
        Pipeline stage name, e.g. ``"displacement"``, ``"done"``.
    progress:
        Percentage complete (0–100).
    message:
        Human-readable status string, e.g. ``"Frame 3/10"``.
    """
    payload = {
        "job_id": job_id,
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await manager.broadcast(job_id, payload)
