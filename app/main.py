from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import aeo, fanout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up heavy models on startup so the first request isn't slow.
    spaCy models are downloaded automatically if not already installed.
    """
    logger.info("AEGIS API starting up — warming NLP models…")
    try:
        from app.services.aeo_checks.direct_answer import _get_nlp

        _get_nlp()
        logger.info("spaCy model loaded ✓")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load spaCy model even after attempted download: %s", exc)
    yield
    logger.info("AEGIS API shutting down.")


app = FastAPI(
    title="AEGIS — Answer Engine & Generative Intelligence Suite",
    description=(
        "AI-powered content intelligence platform that scores, diagnoses, "
        "and improves content for AEO and GEO."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(aeo.router, prefix="/api/aeo", tags=["AEO"])
app.include_router(fanout.router, prefix="/api/fanout", tags=["Fan-Out"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "AEGIS API"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
