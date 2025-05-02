import uuid
from fastapi import Header, APIRouter, HTTPException, status
from pydantic import BaseModel, validator
from typing import Literal, Dict, Any, Optional
from datetime import datetime
import requests
import json
from inference_only_pipeline_v2 import process_user_query

router = APIRouter()


# --- Schemas ---
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ThresholdRequest(BaseModel):
    type: Literal["quiz", "exam"]
    enrollment_id: str
    item_guid: str
    course_guid: str


# --- Helper Functions ---
def fetch_quiz_statistics(
    enrollment_id: str, item_guid: str, token: str
) -> Dict[str, Any]:
    """Fetch quiz statistics from external API."""
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_STATISTICS"

    payload = {
        "assessment_type": "QuizAssessmentResultStatistic",
        "enrollment_id": enrollment_id,
        "item_guid": item_guid,
    }

    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url, json={"json": json.dumps(payload)}, headers=headers
        )
        response.raise_for_status()
        raw_data = response.json()

        data = json.loads(raw_data)

        if isinstance(data, list) and data:
            last_attempt = data[-1]
            total_attempts = len(data)
            total_failed_attempts = sum(
                1
                for attempt in data
                if not attempt.get("ACHIEVEDASSESSMENTMASTERYTF", True)
            )

            last_attempt["TOTALATTEMPTS"] = total_attempts
            last_attempt["TOTALFAILEDATTEMPTS"] = total_failed_attempts

            return last_attempt

        print(f"Unexpected or empty response format: {data}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No quiz statistics found."
        )
    except requests.RequestException as e:
        print(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quiz statistics: {str(e)}",
        )


def fetch_quiz_details(learner_static_id: str, token: str) -> Dict[str, Any]:
    """Simulate fetching quiz details (questions/answers) from external API."""

    url = f"https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_ATTEMPTED_FOR_REVIEW"

    payload = {
        "LEARNERSTATISTIC_ID": learner_static_id,
    }

    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            url, json={"json": json.dumps(payload)}, headers=headers
        )
        response.raise_for_status()
        raw_data = response.json()
        data = json.loads(raw_data)
        print(f"Response from quiz details API: {data}")
        return data
    except requests.RequestException as e:
        print(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quiz details: {str(e)}",
        )


def fetch_exam_statistics(enrollment_id: str, item_guid: str) -> Dict[str, Any]:
    """Simulate fetching exam statistics from external API."""
    # Example based on 4th/5th images
    url = f"https://api.example.com/exam/statistics?enrollment_id={enrollment_id}&item_guid={item_guid}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Response from exam statistics API: {response.json()}")
        return response.json()

    except requests.RequestException as e:
        print(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch exam statistics: {str(e)}",
        )


def fetch_exam_details(enrollment_id: str, item_guid: str) -> Dict[str, Any]:
    """Simulate fetching exam details (questions/answers) from external API."""
    # Example based on 5th image
    url = f"https://api.example.com/exam/details?enrollment_id={enrollment_id}&item_guid={item_guid}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch exam details: {str(e)}",
        )


def evaluate_thresholds(
    data: Dict[str, Any], details: Dict[str, Any], type: str
) -> Dict[str, Any]:
    """Generate a warm and analytical feedback prompt for the LLM."""

    total_attempts = data.get("TOTALATTEMPTS", 0)
    failed_attempts = data.get("TOTALFAILEDATTEMPTS", 0)
    score = data.get("RAWSCORE", 0.0)
    questions_correct = data.get("TOTALQUESTIONSCORRECT", 0)
    questions_incorrect = data.get("TOTALQUESTIONSINCORRECT", 0)
    total_questions = questions_correct + questions_incorrect
    time_spent = data.get("TOTALQUESTIONSCORRECTATTHEENDOFTHEASSESSMENT", 30)

    thresholds_info = []
    if type == "quiz":
        if failed_attempts >= 2:
            thresholds_info.append(
                "Student attempted this quiz multiple times, which indicates some difficulty grasping the material."
            )
        if score < 60:
            thresholds_info.append(
                "Student's score was below 60%, suggesting a need for further review."
            )
        if time_spent < 30:
            thresholds_info.append(
                "Student spent relatively little time on this quiz, which may have impacted performance."
            )
    elif type == "exam":
        if failed_attempts >= 1:
            thresholds_info.append(
                "Student re-attempted the exam, possibly indicating areas that were unclear."
            )
        if score < 60:
            thresholds_info.append(
                "Student's exam score was below 60%, which is below the expected level."
            )
        if time_spent < 120:
            thresholds_info.append(
                "Student spent less than 2 hours on the exam, which might suggest rushing or lack of focus."
            )

    # Process questions
    questions_feedback = []
    incorrect_summary = []
    questions = details if isinstance(details, list) else details.get("questions", [])
    for idx, question in enumerate(questions, 1):
        is_correct = question.get("ANSWEREDCORRECTLY", False)
        question_text = question.get("QUESTIONSTEM", "Question")
        question_type = question.get("QUESTIONTYPE", "N/A")
        user_answer = question.get("ANSWER_SELECTED", "N/A")
        correct_answer = next(
            (
                a["label"]
                for a in question.get("ASSESSMENTITEMANSWER", [])
                if a.get("ISCORRECTTF")
            ),
            "Not provided",
        )

        if not is_correct:
            questions_feedback.append(
                f'{idx}. "{question_text}"\n'
                f"   - Your Answer: {user_answer}\n"
                f"   - Correct Answer: {correct_answer}\n"
                f"   - Question Type: {question_type}\n"
            )
            incorrect_summary.append(
                {
                    "question": question_text,
                    "your_answer": user_answer,
                    "correct_answer": correct_answer,
                }
            )

    intro = (
        f"Hello! Please act like a friendly tutor. Based on the student's performance below, analyze their weaknesses and give helpful, topic-specific guidance.\n"
        f"Include:\n"
        f"- Areas to focus on\n"
        f"- Slides or course topics to revisit\n"
        f"- Friendly encouragement\n\n"
        f"Respond conversationally and avoid repeating raw stats.\n\n"
        f"---\n\n"
        f"Student recently attempted a **{type}**. Here's their performance breakdown:\n\n"
    )

    performance_summary = f"**Summary:**\n- Score: {score:.2f}%\n- Time Spent: {time_spent} minutes\n- Questions Correctly Answered: {questions_correct}\n- Questions Incorrectly Answered: {questions_incorrect}\n- Total Questions: {total_questions}\n- Attempt No.: {total_attempts}\n\n"

    thresholds_section = (
        "**Observations:**\n"
        + "\n".join(f"- {item}" for item in thresholds_info)
        + "\n"
        if thresholds_info
        else ""
    )

    questions_section = (
        "**Incorrect Questions:**\n" + "\n".join(questions_feedback) + "\n"
        if questions_feedback
        else "All questions answered correctly. 🎉"
    )

    llm_query = intro + performance_summary + thresholds_section + questions_section

    return {
        "llm_query": llm_query,
        "data": {
            "score": score,
            "time_spent": time_spent,
            "attempts": failed_attempts,
            "incorrect_questions": incorrect_summary,
        },
    }


# --- API Endpoint ---
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse

security = HTTPBearer()


@router.post("/evaluate-thresholds", response_model=StandardResponse)
async def get_feedback(
    request: ThresholdRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> StreamingResponse:
    try:
        token = credentials.credentials
        print(f"Token received: {token}")
        # Fetch data based on type (quiz or exam)
        if request.type == "quiz":
            stats = fetch_quiz_statistics(
                request.enrollment_id, request.item_guid, token
            )
            details = fetch_quiz_details(stats["ID"], token)
        else:  # exam
            stats = fetch_exam_statistics(request.enrollment_id, request.item_guid)
            details = fetch_exam_details(request.enrollment_id, request.item_guid)

        # Evaluate against thresholds
        feedbackQuery = evaluate_thresholds(stats, details, request.type)
        print(f"Feedback query: {feedbackQuery["llm_query"]}")
        response_generator = process_user_query(
            feedbackQuery["llm_query"], conversation_history=[]
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

        return StarletteStreamingResponse(
            generate(),
            media_type="text/plain",
            # headers={"x-session-id": str(session_id)},
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
