"""
FaultTrace-RAG FastAPI Application.

Versioned REST API with OpenAPI documentation, health checks,
structured logging, CORS, pagination, and error models.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from faulttrace_api.config import get_settings
from faulttrace_api.database import init_db

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    settings = get_settings()
    logger.info("faulttrace_api.startup", version="0.1.0", data_root=str(settings.data_root))
    init_db()
    yield
    logger.info("faulttrace_api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FaultTrace-RAG API",
        description=(
            "Counterfactual Fault Localization for Corpus-Level LLM Analytics Pipelines. "
            "Provides deterministic corpus data, procedural queries, dual gold evaluation, "
            "and traced pipeline execution."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    origins = settings.cors_origins.split(",") if settings.cors_origins else ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-MS"] = f"{duration_ms:.1f}"
        return response

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # Include routers
    from faulttrace_api.routes import health, system, worlds, queries, gold, runs, demo, artifacts
    from faulttrace_api.routes import datasets, query_packs, disagreements, providers, policies

    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(system.router, prefix="/api/v1", tags=["System"])
    app.include_router(demo.router, prefix="/api/v1", tags=["Demo"])
    app.include_router(worlds.router, prefix="/api/v1", tags=["Worlds"])
    app.include_router(queries.router, prefix="/api/v1", tags=["Queries"])
    app.include_router(gold.router, prefix="/api/v1", tags=["Gold"])
    app.include_router(runs.router, prefix="/api/v1", tags=["Runs"])
    app.include_router(artifacts.router, prefix="/api/v1", tags=["Artifacts"])
    # Prompt 2: data quality UI/API
    app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
    app.include_router(query_packs.router, prefix="/api/v1", tags=["QueryPacks"])
    app.include_router(disagreements.router, prefix="/api/v1", tags=["Disagreements"])
    app.include_router(providers.router, prefix="/api/v1", tags=["Providers"])
    app.include_router(policies.router, prefix="/api/v1", tags=["Policies"])

    return app


app = create_app()
