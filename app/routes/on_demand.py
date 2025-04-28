"""
CODE For only chatting with groq inference and gui , upserting code has all been removed

"""

import os
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
import requests
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from pinecone import Pinecone
from sqlalchemy.orm import Session
from models.chat import ChatMessage, ChatSession
from inference_only_pipeline_v2 import process_user_query
from schemas.chat import (
    OnDemandReqBody,
    StandardResponse,
    PaginatedStandardResponse,
    ChatMessageSchema,
    ChatSessionSchema,
)
from database.connections import get_db

load_dotenv()

router = APIRouter()

from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR


from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List

from fastapi import Query
import math


@router.post("/process", response_model=StandardResponse)
async def process_query(
    request: OnDemandReqBody, db: Session = Depends(get_db)
) -> StandardResponse:
    try:
        # Check if user_query is missing or empty
        if not request.user_query or not isinstance(request.user_query, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User query is required and must be a string.",
            )
        # If session_id is not provided, create a new ChatSession
        if not request.session_id:
            new_session = ChatSession(
                user_id=request.user_id, created_at=datetime.utcnow()
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            session_id = new_session.id
        else:
            session_id = request.session_id

        # Save user message
        user_message = ChatMessage(
            session_id=session_id, sender="user", content=request.user_query
        )
        db.add(user_message)
        db.commit()

        # Process the user's query
        response_generator = process_user_query(request.user_query, conversation_history=[])
        
        # Iterate through the generator to get the final history
        final_history = None
        context = None
        for history, ctx in response_generator:
            final_history = history
            context = ctx
        
        # Extract the bot's response from the final history
        if not final_history or len(final_history) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No response generated from the query.",
            )
        
        # The last tuple in final_history contains (user_query, bot_response)
        response_text = final_history[-1][1]  # Get the bot's response
        print("Response Text:", response_text)

        # Save bot response
        bot_message = ChatMessage(
            session_id=session_id, sender="bot", content=response_text
        )
        db.add(bot_message)
        db.commit()

        return StandardResponse(
            status_code=status.HTTP_200_OK,
            message="Success",
            data={"session_id": session_id, "response": response_text},
        )

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
    if not user_id:
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
