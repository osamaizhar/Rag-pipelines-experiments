from typing import Any, Optional
from pydantic import BaseModel

class StandardResponse(BaseModel):
    statusCode: int
    message: str
    data: Optional[Any] = None


class OnDemandReqBody(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    user_query: str


class FailureStatisticsRequest(BaseModel):
    token: str
    enrollment_id: str
    item_guid: str
    type: str  # 'quiz' or 'exam'
