from typing import Any, Optional
from pydantic import BaseModel, Field, validator
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

    # Validator to ensure that session_id or user_id is provided when necessary (e.g., if a session ID is present, user_id should be provided)
    @validator('user_query')
    def user_query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('User query must not be empty')
        return v


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
