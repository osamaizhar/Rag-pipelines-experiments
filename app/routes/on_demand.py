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
from schemas.chat import (
    OnDemandReqBody,
    StandardResponse,
    PaginatedStandardResponse,
    ChatMessageSchema,
    ChatSessionSchema,
)
from database.connections import get_db

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
PINECONE_API = os.getenv("PINECONE_API")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("PINECONE_API", PINECONE_API)


# Groq API settings
GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL = "llama3-405b-8192-embed"
LLM_MODEL = "llama3-70b-8192"


# Configure headers for Groq API requests
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

pc = Pinecone(api_key=PINECONE_API)
print(PINECONE_API)

index = pc.Index("ai-coach")
# index = pc.Index("ahsan-400pg-pdf-doc-test")


embedding_model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
)

# Function to generate embeddings without tokenization


def get_embedding(data):
    embeddings = embedding_model.encode(data).tolist()
    return embeddings


# Function to query Pinecone index using embeddings
def query_pinecone(embedding):
    # Use keyword arguments to pass the embedding and other parameters
    result = index.query(vector=embedding, top_k=15, include_metadata=True)
    return result["matches"]


# Function to query Groq LLM
def query_groq(prompt: str) -> str:
    response = requests.post(
        GROQ_CHAT_URL,
        headers=GROQ_HEADERS,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 8192,  # max from groq website
        },
    )

    if response.status_code != 200:
        raise Exception(f"Error querying Groq: {response.text}")
    return response.json()["choices"][0]["message"]["content"]


# Tokenizer to count number of tokens
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")


def count_tokens(text: str) -> int:
    # Encode the text into tokens
    tokens = tokenizer.encode(text)
    return len(tokens)


# ------------------------------------------- Main Function -------------------------------------------
# Initialize empty conversation history (list of tuples)
conversation_history = []


def process_user_query(user_query: str, conversation_history: list):
    print(f"User Query Tokens: {count_tokens(user_query)}")

    # Generate embedding and get relevant context
    embedding = get_embedding(user_query)
    relevant_chunks = query_pinecone(embedding)
    context = "\n".join(chunk["metadata"]["text"] for chunk in relevant_chunks)
    print("CONTEXT:", context)

    # Format conversation history for the prompt
    history_str = "\n".join(
        f"User: {user}\nCoach: {response}" for user, response in conversation_history
    )

    # Create structured prompt
    prompt = f"""You are a knowledgeable and friendly coach. Follow these guidelines:
    1. Provide clear, step-by-step explanations
    2. Ask guiding questions to encourage critical thinking
    3. Adapt to the student's knowledge level
    4. Use examples from the provided context when relevant

    Context from learning materials:
    {context}

    Conversation history:
    {history_str}

    New student question:
    "{user_query}"

    Provide a helpful response:"""

    # Get LLM response
    groq_response = query_groq(prompt)
    print(f"Response Tokens: {count_tokens(groq_response)}")

    # Return updated history with new interaction
    return conversation_history + [(user_query, groq_response)]


# Gradio Interface
def start_gradio():
    with gr.Blocks() as interface:
        gr.Markdown("# 🧑‍🏫 AI Coaching Assistant")
        gr.Markdown("Welcome! I'm here to help you learn. Type your question below.")

        # State management
        chat_history = gr.State(conversation_history)

        with gr.Row():
            chatbot = gr.Chatbot(height=500)
            with gr.Column(scale=0.5):
                context_display = gr.Textbox(
                    label="Relevant Context", interactive=False
                )

        user_input = gr.Textbox(label="Your Question", placeholder="Type here...")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            undo_btn = gr.Button("Undo Last")
            clear_btn = gr.Button("Clear History")

        def handle_submit(user_input, history):
            if not user_input.strip():
                return gr.update(), history, ""

            # Process query and update history
            new_history = process_user_query(user_input, history)

            # Get latest context for display
            latest_context = "\n".join(
                [
                    chunk["metadata"]["text"]
                    for chunk in query_pinecone(get_embedding(user_input))
                ][:3]
            )  # Show top 3 context snippets

            return "", new_history, latest_context

        # Component interactions
        submit_btn.click(
            handle_submit,
            [user_input, chat_history],
            [user_input, chat_history, context_display],
        ).then(lambda x: x, [chat_history], [chatbot])

        # Add submit on Enter key press
        user_input.submit(
            handle_submit,
            [user_input, chat_history],
            [user_input, chat_history, context_display],
        ).then(lambda x: x, [chat_history], [chatbot])

        undo_btn.click(
            lambda history: history[:-1] if history else [],
            [chat_history],
            [chat_history],
        ).then(lambda x: x, [chat_history], [chatbot])

        clear_btn.click(lambda: [], None, [chat_history]).then(
            lambda: ([], ""), None, [chatbot, context_display]
        )

    # interface.launch(share=True)
    interface.launch(server_name="0.0.0.0", share=True)


router = APIRouter()

from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR


from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List


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
        response_pairs = process_user_query(request.user_query, conversation_history=[])
        response_text = response_pairs[0][1]
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


from fastapi import Query
import math


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
