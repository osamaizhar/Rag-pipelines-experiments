from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse
from pydantic import BaseModel
from typing import Literal, Dict, Any, Optional, List
import httpx
import json
import asyncio
from sqlalchemy.orm import Session
from database.connections import get_db
from inference_only_pipeline_v2 import process_user_query
from datetime import datetime
from models.chat import ChatMessage, ChatSession

router = APIRouter()
security = HTTPBearer()


# --- Schemas ---
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ThresholdRequest(BaseModel):
    type: Literal["quiz", "exam"]
    enrollment_id: str
    item_guid: Optional[str] = None
    course_guid: Optional[str] = None
    time_taken: Optional[float] = None
    name: Optional[str] = None
    bootcamp_name: Optional[str] = None
    user_id: str


# --- Utility ---
def build_headers(token: str) -> dict:
    return {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# --- External API Calls ---
async def safe_json_response(response):
    try:
        raw = response.json()
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


async def fetch_quiz_statistics(
    enrollment_id: str, item_guid: str, token: str
) -> Dict[str, Any]:
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_STATISTICS"
    payload = {
        "assessment_type": "QuizAssessmentResultStatistic",
        "enrollment_id": enrollment_id,
        "item_guid": item_guid,
    }
    headers = build_headers(token)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url, json={"json": json.dumps(payload)}, headers=headers
        )
        response.raise_for_status()
        data = await safe_json_response(response)

        if isinstance(data, list) and data:
            return {
                "data": data,
                "stats": {
                    "TOTALATTEMPTS": len(data),
                    "TOTALFAILEDATTEMPTS": sum(
                        not attempt.get("ACHIEVEDASSESSMENTMASTERYTF", True)
                        for attempt in data
                    ),
                },
            }
        raise HTTPException(status_code=404, detail="No quiz statistics found.")


async def fetch_exam_statistics(enrollment_id: str, token: str) -> Dict[str, Any]:
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_STATISTICS"
    payload = {
        "assessment_type": "PostAssessmentResultStatistic",
        "enrollment_id": enrollment_id,
    }
    headers = build_headers(token)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url, json={"json": json.dumps(payload)}, headers=headers
        )
        response.raise_for_status()
        data = await safe_json_response(response)

        if isinstance(data, list) and data:
            return {
                "data": data,
                "stats": {
                    "TOTALATTEMPTS": len(data),
                    "TOTALFAILEDATTEMPTS": sum(
                        not attempt.get("ACHIEVEDASSESSMENTMASTERYTF", True)
                        for attempt in data
                    ),
                },
            }
        raise HTTPException(status_code=404, detail="No quiz statistics found.")


async def fetch_assessment_details(
    learner_stat_id: str, token: str
) -> List[Dict[str, Any]]:
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_ATTEMPTED_FOR_REVIEW"
    payload = {"LEARNERSTATISTIC_ID": learner_stat_id}
    headers = build_headers(token)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url, json={"json": json.dumps(payload)}, headers=headers
        )
        response.raise_for_status()
        data = await safe_json_response(response)
        return data or []


# --- Evaluation ---
def evaluate_thresholds(
    all_data: list,
    type_: str,
    name=None,
    user_id: str = None,
    bootcamp_name: str = None,
) -> Dict[str, Any]:
    llm_query = (
        f"Hello! You're a friendly tutor. Based on the student's **{type_}**, "
        "analyze their weaknesses and provide topic-specific feedback:\n"
        "- Areas to improve\n"
        "- Topics to revisit\n"
        "- Encouragement\n\n"
        "---\n\n"
    )

    if bootcamp_name:
        llm_query += f"- Bootcamp Name: {bootcamp_name}\n"

    if name and type_ == "exam":
        llm_query += f"- Course Name: {name}\n"
    elif name and type_ == "quiz":
        llm_query += f"- Lesson Name: {name}\n"

    if user_id:
        llm_query += f"- Student ID: {user_id}\n"

    questions_list = []

    for idx, data in enumerate(all_data, 1):
        score = data.get("RAWSCORE", 0)
        correct = data.get("TOTALQUESTIONSCORRECT", 0)
        incorrect = data.get("TOTALQUESTIONSINCORRECT", 0)
        total = correct + incorrect
        time_spent = data.get("TOTALTIMESPENTINSECONDSASSESSMENT", 0) / 60
        time_spent = round(time_spent, 2) if time_spent else "N/A"
        date = data.get("STATISTICDATE", "N/A")

        llm_query += (
            f"---\n\n"
            f"**Attempt {idx}**\n"
            f"- Score: {score:.2f}%\n"
            f"- Date: {date}\n"
            f"- Time Spent: {time_spent} minutes\n"
            f"- Correct: {correct}\n"
            f"- Incorrect: {incorrect}\n"
            f"- Total: {total}\n\n"
        )

        for q in data.get("details", []):
            question_text = q.get("QUESTIONSTEM", "N/A")
            user_answer = q.get("ANSWER_SELECTED", "N/A")
            question_id = q.get("ASSESSMENTITEMGUID", "N/A")
            correct_answer = next(
                (
                    a["label"]
                    for a in q.get("ASSESSMENTITEMANSWER", [])
                    if a.get("ISCORRECTTF")
                ),
                "Not provided",
            )
            llm_query += f'- Question:  "{question_text}"\n  - Student Answer: {user_answer}\n  - Correct Answer: {correct_answer}\n  - Question ID: {question_id}\n\n'
            questions_list.append(
                {
                    "question": question_text,
                    "your_answer": user_answer,
                    "correct_answer": correct_answer,
                }
            )

        llm_query += "\n"

    return {
        "llm_query": llm_query,
        "data": {
            "score": score,
            "time_spent": time_spent,
            "attempts": len(all_data),
            "questions": questions_list,
        },
    }


# --- Endpoint ---
@router.post("/evaluate-thresholds")
async def get_feedback(
    request: ThresholdRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    token = credentials.credentials

    try:
        if not request.user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID must be provided.",
            )
        if request.type == "quiz":
            if not request.enrollment_id or not request.item_guid:
                raise HTTPException(
                    400, "Enrollment ID and Item GUID are required for quizzes."
                )

            quiz_data = await fetch_quiz_statistics(
                request.enrollment_id, request.item_guid, token
            )
            attempts = quiz_data["data"]
            detail_tasks = [fetch_assessment_details(a["ID"], token) for a in attempts]
            details = await asyncio.gather(*detail_tasks)

            for a, d in zip(attempts, details):
                a["details"] = d

            feedback_query = evaluate_thresholds(
                attempts,
                request.type,
                request.name,
                request.user_id,
                request.bootcamp_name,
            )

        elif request.type == "exam":
            if not request.enrollment_id:
                raise HTTPException(400, "Enrollment ID is required for exams.")

            exam_data = await fetch_exam_statistics(request.enrollment_id, token)
            attempts = exam_data["data"]
            detail_tasks = [fetch_assessment_details(a["ID"], token) for a in attempts]
            details = await asyncio.gather(*detail_tasks)
            for a, d in zip(attempts, details):
                a["details"] = d

            feedback_query = evaluate_thresholds(
                attempts,
                request.type,
                request.name,
                request.user_id,
                request.bootcamp_name,
            )

        else:
            raise HTTPException(400, "Invalid type. Must be 'quiz' or 'exam'.")

        
        print(f"Feedback query: {feedback_query["llm_query"]}")
        print(f"\nQuery length: {len(feedback_query['llm_query'])} chars")
        if request.name is not None and not request.name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name cannot be empty.",
            )

        # Create new session
        new_session = ChatSession(user_id=request.user_id, created_at=datetime.utcnow())
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        session_id = new_session.id

        async def generate():
            previous = ""
            try:
                response_generator = process_user_query(
                    feedback_query["llm_query"], conversation_history=[]
                )
                for updated_history, _ in response_generator:
                    if updated_history:
                        current = updated_history[-1][1]
                        delta = current[len(previous) :]
                        previous = current
                        if delta:
                            yield delta.encode("utf-8")

                bot_message = ChatMessage(
                    session_id=session_id, sender="bot", content=previous
                )
                db.add(bot_message)
                db.commit()
            except Exception as e:
                # Return LLM processing error details
                yield f"\n[Error during response generation: {str(e)}]".encode("utf-8")

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
