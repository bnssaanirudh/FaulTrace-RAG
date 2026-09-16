"""
FaultTrace-RAG FastAPI Application.

Versioned REST API with OpenAPI documentation, health checks,
structured logging, CORS, pagination, and error models.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

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
            "traced pipeline execution, evidence-grounded extraction with citation integrity, "
            "and provenance knowledge graph construction."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    origins = (
        settings.cors_origins.split(",") if settings.cors_origins else ["http://localhost:3000"]
    )
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

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": str(exc.detail),
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc) if settings.debug else "An internal error occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # Include routers
    from faulttrace_api.routes import (
        analytics,
        annotations,
        artifacts,
        datasets,
        demo,
        disagreements,
        experiments,
        gold,
        governance,
        health,
        policies,
        providers,
        queries,
        query_packs,
        retrieval,
        runs,
        system,
        worlds,
    )

    # Starlette resolves the first matching route. Keep literal endpoints such
    # as /datasets/text and /runs/batch-evaluate-policy ahead of parameterized
    # routes such as /datasets/{snapshot_id} and /runs/{run_id}.
    def route_specificity(route) -> tuple[int, int, str]:
        path = getattr(route, "path", "")
        return (path.count("{"), -len(path.split("/")), path)

    for module in (
        health,
        system,
        demo,
        worlds,
        queries,
        gold,
        runs,
        artifacts,
        datasets,
        query_packs,
        disagreements,
        providers,
        policies,
        experiments,
        annotations,
        governance,
        retrieval,
        analytics,
    ):
        module.router.routes.sort(key=route_specificity)

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
    app.include_router(experiments.router, prefix="/api/v1", tags=["Experiments"])
    app.include_router(annotations.router, prefix="/api/v1", tags=["Annotations"])
    app.include_router(governance.router, prefix="/api/v1", tags=["Governance"])
    # Prompt 3: Real retrieval and analytics
    app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
    # Prompt 4: Evidence extraction, citation integrity, provenance graph
    from faulttrace_api.routes import evidence

    app.include_router(evidence.router, prefix="/api/v1/evidence", tags=["evidence"])

    return app


app = create_app()
