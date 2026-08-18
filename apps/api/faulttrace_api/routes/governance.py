from uuid import uuid4

from fastapi import APIRouter, Request
from faulttrace_core.governance import AuditLog

router = APIRouter(prefix="/governance", tags=["governance"])

_MOCK_AUDIT_LOGS: list[AuditLog] = []


@router.get("/audit", response_model=list[AuditLog])
async def list_audit_logs():
    return _MOCK_AUDIT_LOGS


@router.post("/audit", response_model=AuditLog)
async def create_audit_log(
    request: Request, user_id: str, action: str, resource_type: str, resource_id: str, details: dict
):
    log = AuditLog(
        log_id=f"audit_{uuid4().hex[:8]}",
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=request.client.host if request.client else None,
    )
    _MOCK_AUDIT_LOGS.append(log)
    return log
