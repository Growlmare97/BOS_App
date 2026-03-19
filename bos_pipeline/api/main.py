"""FastAPI application entry point for the BOS Analysis backend.

Start with:
    uvicorn bos_pipeline.api.main:app --reload --port 8000

Or via:
    bos_server  (entry point in pyproject.toml)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from bos_pipeline.api import routes, websocket

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BOS Analysis API",
    description=(
        "Background-Oriented Schlieren image processing backend. "
        "Supports Photron and DALSA camera formats, cross-correlation and "
        "optical-flow displacement methods, Abel inversion, gas-concentration "
        "measurement and velocity estimation."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server and any localhost origin during development
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server (default port)
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # CRA / other common dev ports
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    logger.info("BOS Analysis API ready")


@app.on_event("shutdown")
async def _shutdown() -> None:
    logger.info("BOS Analysis API shutting down")


# ---------------------------------------------------------------------------
# Health check  (must be registered before any catch-all static mount)
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe — returns ``{"status": "ok", "version": "0.1.0"}``."""
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Routers  (also before static mount)
# ---------------------------------------------------------------------------

# REST endpoints — all live under /api
app.include_router(routes.router, prefix="/api")

# WebSocket endpoint — /ws/{job_id}
app.include_router(websocket.router)

# ---------------------------------------------------------------------------
# Static frontend — mounted LAST so API routes take priority
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )
    logger.info("Serving frontend from %s", _FRONTEND_DIST)


# ---------------------------------------------------------------------------
# CLI entry point (bos_server script)
# ---------------------------------------------------------------------------


def start() -> None:
    """Start the BOS API server (used by ``bos_server`` CLI entry point)."""
    import uvicorn

    uvicorn.run(
        "bos_pipeline.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
