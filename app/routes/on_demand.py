"""
CODE For only chatting with groq inference and gui , upserting code has all been removed

"""

import uuid
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
import gradio as gr
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from models.chat import ChatMessage, ChatSession
from on_demand_coaching_beta_v1 import process_user_query
from schemas.chat import (
    OnDemandReqBody,
    StandardResponse,
    PaginatedStandardResponse,
    ChatMessageSchema,
    ChatSessionSchema,
)
from database.connections import get_db
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from fastapi import HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
import math


load_dotenv()

router = APIRouter()


@router.post("/process")
async def process_query(
    request: OnDemandReqBody, db: Session = Depends(get_db)
) -> StreamingResponse:
    try:
        start_time = time.perf_counter()
        if not request.user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="User ID must be provided."
            )
        if not request.user_query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_query cannot be empty.",
            )

        # Validate and fetch/create session
        if request.session_id is not None:
            if not request.session_id.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Session Id cannot be empty.",
                )
            try:
                uuid.UUID(request.session_id)
            except ValueError as e:
                print(f"Error fetching messages: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Session ID must be a valid UUID.",
                )
            session = (
                db.query(ChatSession)
                .filter(ChatSession.id == request.session_id)
                .first()
            )
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found.",
                )
            session_id = session.id
            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(10)
                .all()
            )

            message_data = [ChatMessageSchema.model_validate(m) for m in messages]
        else:
            new_session = ChatSession(
                user_id=request.user_id, created_at=datetime.utcnow()
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            session_id = new_session.id

        # Save user message
        user_message = ChatMessage(
            session_id=session_id, sender="user", content=request.user_query
        )
        db.add(user_message)
        db.commit()

        # Process and stream response
        response_generator = process_user_query(
            request.user_query, conversation_history=[]
        )

        async def generate():
            previous = ""
            bot_full_message = ""
            for updated_history, context in response_generator:
                if updated_history:
                    current_full = updated_history[-1][1]
                    new_part = current_full[len(previous) :]
                    previous = current_full
                    bot_full_message = current_full
                    if new_part:
                        yield new_part

            # Save bot response
            bot_message = ChatMessage(
                session_id=session_id, sender="bot", content=bot_full_message
            )
            db.add(bot_message)
            db.commit()

        elapsed = time.perf_counter() - start_time
        print(f"⏱ Time until stream starts: {elapsed:.2f} seconds")

        return StarletteStreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"x-session-id": str(session_id)},
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get("/sessions/{user_id}", response_model=PaginatedStandardResponse)
def get_sessions_by_user(
    user_id: str,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Records per page"),
) -> StandardResponse:
    if not user_id and not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User ID must be provided."
        )

    try:
        total = db.query(ChatSession).filter(ChatSession.user_id == user_id).count()
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        sessions_data = [ChatSessionSchema.model_validate(s) for s in sessions]

        return PaginatedStandardResponse(
            status_code=status.HTTP_200_OK,
            message="Success",
            data=sessions_data,
            page=page,
            limit=limit,
            total=total,
            last_page=math.ceil(total / limit) if limit else 1,
        )

    except Exception as e:
        print(f"Error fetching sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get("/sessions/{session_id}/messages", response_model=PaginatedStandardResponse)
def get_messages_by_session(
    session_id: str,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=1000, description="Records per page"),
) -> StandardResponse:

    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID must be provided.",
        )
    try:
        uuid.UUID(session_id)
    except ValueError as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID must be a valid UUID.",
        )
    try:
        total = (
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
        )
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        message_data = [ChatMessageSchema.model_validate(m) for m in messages]

        return PaginatedStandardResponse(
            status_code=status.HTTP_200_OK,
            message="Messages fetched successfully",
            data=message_data,
            page=page,
            limit=limit,
            total=total,
            last_page=math.ceil(total / limit) if limit else 1,
        )

    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
