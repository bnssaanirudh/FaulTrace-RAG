from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditLog(BaseModel):
    log_id: str
    user_id: str
    action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resource_type: str
    resource_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
