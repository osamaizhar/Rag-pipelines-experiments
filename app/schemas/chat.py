from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID


class StandardResponse(BaseModel):
    statusCode: int
    message: str
    data: Optional[Any] = None


class OnDemandReqBody(BaseModel):
    user_id: str = Field(..., title="User ID", description="The ID of the user making the request")
    session_id: Optional[str] = Field(None, title="Session ID", description="The ID of the session (optional)")
    user_query: str = Field(..., title="User Query", description="The query submitted by the user")


class FailureStatisticsRequest(BaseModel):
    token: str
    enrollment_id: str
    item_guid: str
    type: str  # 'quiz' or 'exam'




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
