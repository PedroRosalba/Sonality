"""Local-only FastAPI wrapper around SonalityAgent.

This file is a local development tool for testing the Risklayer <-> Sonality
integration end-to-end. It is **not** part of the Sonality package deliverable
and should not be committed to the Sonality repo.

Usage:
    uv add fastapi uvicorn  # one-time, local only
    uv run uvicorn sonality.server:app --port 8100

Contract (see daloc-risklayer/SYSTEM_DESIGN.md):
    POST /respond       { "message": "..." } -> { "response": "...", "ess": {...} }
    GET  /health        -> { "status": "ok", "interaction_count": N, ... }
    GET  /personality   -> { "snapshot": "...", "beliefs": {...}, "version": N }
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import SonalityAgent

log = logging.getLogger(__name__)


# ─── Request/response schemas ───────────────────────────────────────────────


class RespondRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)


class ESSPayload(BaseModel):
    score: float
    reasoning_type: str
    opinion_direction: str
    topics: list[str]
    novelty: float


class RespondResponse(BaseModel):
    response: str
    ess: ESSPayload


class HealthResponse(BaseModel):
    status: str
    interaction_count: int
    belief_count: int
    version: int


class PersonalityResponse(BaseModel):
    snapshot: str
    beliefs: dict[str, float]
    version: int


# ─── Lifespan: instantiate the agent once and keep it warm in memory ────────


_agent: SonalityAgent | None = None
# SonalityAgent is single-threaded (maintains mutable sponge state across
# respond() calls). Serialize concurrent requests with a lock so FastAPI's
# default worker model doesn't interleave turns.
_agent_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    global _agent
    log.info("Booting SonalityAgent for HTTP server…")
    _agent = SonalityAgent()
    log.info(
        "SonalityAgent ready: v%d, %d interactions, %d beliefs",
        _agent.sponge.version,
        _agent.sponge.interaction_count,
        len(_agent.sponge.opinion_vectors),
    )
    try:
        yield
    finally:
        log.info("Shutting down SonalityAgent…")
        if _agent is not None:
            try:
                _agent.shutdown()
            except Exception:
                log.exception("Agent shutdown raised")
        _agent = None


app = FastAPI(title="Sonality (local dev server)", lifespan=lifespan)


def _require_agent() -> SonalityAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _agent


# ─── POST /respond ──────────────────────────────────────────────────────────
#
# Each call is stateless at the conversation level: we clear the per-instance
# chat buffer before calling respond() so callers can't accidentally leak
# context across independent messages. Long-term memory (Neo4j + pgvector) and
# the sponge state persist as side effects, which is the whole point.


@app.post("/respond", response_model=RespondResponse)
def respond(req: RespondRequest) -> RespondResponse:
    agent = _require_agent()
    with _agent_lock:
        agent.conversation = []
        try:
            reply = agent.respond(req.message)
        except Exception as exc:
            log.exception("Agent.respond failed")
            raise HTTPException(status_code=500, detail=f"respond failed: {exc}") from exc
        ess = agent.last_ess

    return RespondResponse(
        response=reply,
        ess=ESSPayload(
            score=float(ess.score),
            reasoning_type=str(ess.reasoning_type),
            opinion_direction=str(ess.opinion_direction),
            topics=list(ess.topics),
            novelty=float(ess.novelty),
        ),
    )


# ─── GET /health ────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _agent is None:
        return HealthResponse(status="starting", interaction_count=0, belief_count=0, version=0)
    sponge = _agent.sponge
    return HealthResponse(
        status="ok",
        interaction_count=int(sponge.interaction_count),
        belief_count=int(len(sponge.opinion_vectors)),
        version=int(sponge.version),
    )


# ─── GET /personality ───────────────────────────────────────────────────────


@app.get("/personality", response_model=PersonalityResponse)
def personality() -> PersonalityResponse:
    agent = _require_agent()
    sponge = agent.sponge
    # Float-cast each belief value so pydantic doesn't trip on numpy scalars or
    # similar that might sneak in from downstream helpers.
    beliefs: dict[str, float] = {str(k): float(v) for k, v in sponge.opinion_vectors.items()}
    return PersonalityResponse(
        snapshot=str(sponge.snapshot),
        beliefs=beliefs,
        version=int(sponge.version),
    )


# ─── Friendly error handler so risklayer sees a body on 500s ────────────────


@app.exception_handler(HTTPException)
def _http_exception_handler(_request: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
