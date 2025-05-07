from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID


class StandardResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[Any] = None


class PaginatedStandardResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[Any] = None
    page: int
    limit: int
    total: int
    last_page: int


class OnDemandReqBody(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    user_query: str


class FailureStatisticsRequest(BaseModel):
    token: str
    enrollment_id: Optional[str] = None
    item_guid: Optional[str] = None
    type: Literal["quiz", "exam"]


class ChatMessageSchema(BaseModel):
    id: UUID
    session_id: UUID
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True  # Enables SQLAlchemy model to schema conversion


class ChatSessionSchema(BaseModel):
    id: UUID
    user_id: str
    title: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True
