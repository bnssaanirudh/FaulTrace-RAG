"""Health check endpoint."""

from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "service": "faulttrace-api",
        "version": "0.1.0",
        "timestamp": datetime.now(UTC).isoformat(),
    }
