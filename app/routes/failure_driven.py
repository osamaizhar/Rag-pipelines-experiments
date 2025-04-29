import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, validator
from typing import Literal, Dict, Any, Optional
from datetime import datetime
import requests

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
def fetch_quiz_statistics(enrollment_id: str, item_guid: str) -> Dict[str, Any]:
    """Simulate fetching quiz statistics from external API."""
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/global-switch/player/getQuizAssessmentStatistics"
    payload = {
        "enrollmentId": enrollment_id,
        "contentObjectGuid": item_guid
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quiz statistics: {str(e)}"
        )

def fetch_quiz_details(enrollment_id: str, item_guid: str) -> Dict[str, Any]:
    """Simulate fetching quiz details (questions/answers) from external API."""
    # Example based on 3rd image
    url = f"https://api.example.com/quiz/details?enrollment_id={enrollment_id}&item_guid={item_guid}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quiz details: {str(e)}"
        )

def fetch_exam_statistics(enrollment_id: str, item_guid: str) -> Dict[str, Any]:
    """Simulate fetching exam statistics from external API."""
    # Example based on 4th/5th images
    url = f"https://api.example.com/exam/statistics?enrollment_id={enrollment_id}&item_guid={item_guid}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch exam statistics: {str(e)}"
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch exam details: {str(e)}"
        )

def evaluate_thresholds(data: Dict[str, Any], details: Dict[str, Any], type: str) -> Dict[str, Any]:
    """Evaluate the fetched data against thresholds and provide feedback."""
    feedback = {"issues": [], "suggestions": [], "details": []}
    
    # Extract relevant metrics from API response
    failed_attempts = data.get("ATTEMPTNUMBER", 0)
    score = data.get("RAWSCORE", 0.0) / data.get("TOTALQUESTIONSCORRECT", 1) * 100  # Convert to percentage
    time_spent = data.get("TOTALQUESTIONSCORRECTATTHEENDOFTHEASSESSMENT", 0)  # Assuming this is time in minutes
    
    # Thresholds from the first image
    if type == "quiz":
        # Quiz Failed Attempts (>= 2)
        if failed_attempts >= 2:
            feedback["issues"].append(f"Failed quiz attempts: {failed_attempts} (threshold >= 2)")
            feedback["suggestions"].append("Most students have 0 failed attempts. Review the material thoroughly before attempting again.")
        
        # Low Average Quiz Score (< 60%)
        if score < 60:
            feedback["issues"].append(f"Low quiz score: {score:.2f}% (threshold < 60%)")
            feedback["suggestions"].append("Below 60% suggests serious comprehension issues. Focus on understanding key concepts and revisit the lessons.")
        
        # Time Spent on Lesson (< 30 minutes)
        if time_spent < 30:
            feedback["issues"].append(f"Time spent on lesson: {time_spent} minutes (threshold < 30 minutes)")
            feedback["suggestions"].append("AI flags the lesson as possibly skimmed. Spend more time engaging with the content.")
    
    elif type == "exam":
        # Exam Failed Attempts (>= 1)
        if failed_attempts >= 1:
            feedback["issues"].append(f"Failed exam attempts: {failed_attempts} (threshold >= 1)")
            feedback["suggestions"].append("Fewer students fail exams. Even 1 failure should initiate coaching. Consider scheduling a session with an instructor.")
        
        # Low Average Exam Score (< 60%)
        if score < 60:
            feedback["issues"].append(f"Low exam score: {score:.2f}% (threshold < 60%)")
            feedback["suggestions"].append("Consistent with quiz logic: Below 60% indicates comprehension issues. Review core topics and seek clarification.")
        
        # Time Spent on Course (< 2 hours)
        if time_spent < 120:  # Convert 2 hours to minutes
            feedback["issues"].append(f"Time spent on course: {time_spent} minutes (threshold < 120 minutes)")
            feedback["suggestions"].append("AI flags the course as completed too quickly. Dedicate more time to each section for better retention.")
    
    # Analyze correct/incorrect answers from details
    for question in details.get("questions", []):
        question_text = question.get("QUESTIONTEXT", "Unknown question")
        is_correct = question.get("ISCORRECT", False)
        correct_answer = question.get("ANSWER", "N/A")
        user_answer = question.get("USERANSWER", "N/A")
        
        feedback["details"].append({
            "question": question_text,
            "is_correct": is_correct,
            "correct_answer": correct_answer,
            "user_answer": user_answer
        })
        if not is_correct:
            feedback["suggestions"].append(f"Review the concept related to: '{question_text}'. Correct answer: {correct_answer}")

    return feedback

# --- API Endpoint ---
@router.post("/evaluate-thresholds", response_model=StandardResponse)
async def evaluate_thresholds(request: ThresholdRequest):
    try:
        # Fetch data based on type (quiz or exam)
        if request.type == "quiz":
            stats = fetch_quiz_statistics(request.enrollment_id, request.item_guid)
            details = fetch_quiz_details(request.enrollment_id, request.item_guid)
        else:  # exam
            stats = fetch_exam_statistics(request.enrollment_id, request.item_guid)
            details = fetch_exam_details(request.enrollment_id, request.item_guid)

        # Evaluate against thresholds
        feedback = evaluate_thresholds(stats, details, request.type)

        return StandardResponse(
            success=True,
            message="Threshold evaluation completed successfully",
            data=feedback
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )