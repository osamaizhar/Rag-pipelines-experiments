from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Any
import requests
from schemas.chat import FailureStatisticsRequest, StandardResponse

router = APIRouter()


@router.post("/process_failure_statistics", response_model=StandardResponse)
async def process_failure_statistics(data: FailureStatisticsRequest):
    stats_response = get_assessment_statistics(
        data.token, data.enrollment_id, data.item_guid, data.type
    )

    if not stats_response:
        return StandardResponse(
            statusCode=400, message="Failed to fetch assessment statistics", data=None
        )

    coaching_queries = analyze_thresholds(stats_response, data.type)

    if not coaching_queries:
        return StandardResponse(
            statusCode=200, message="No threshold violations found", data=[]
        )

    llm_responses = []
    for query in coaching_queries:
        response = query_llm(query)
        llm_responses.append(response)

    return StandardResponse(
        statusCode=200,
        message="Coaching feedback generated successfully",
        data=llm_responses,
    )


def get_assessment_statistics(
    token: str, enrollment_id: str, item_guid: str, assessment_type: str
):
    url = "https://qa-app.healthtechacademy.org/LS360ApiGateway/services/rest/switch/GET_ASSESSMENT_STATISTICS"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    body = {
        "json": {
            "assessment_type": (
                "QuizAssessmentResultStatistic"
                if assessment_type == "quiz"
                else "ExamAssessmentResultStatistic"
            ),
            "enrollment_id": enrollment_id,
            "item_guid": item_guid,
        }
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def analyze_thresholds(stats_data, assessment_type: str):
    queries = []

    attempts = stats_data[0].get("ASSESSMENTATTEMPTNUMBER", 0)
    raw_score = stats_data[0].get("RAWSCORE", 0)
    mastery_time = stats_data[0].get("MASTERSCOREATTIMEOFASSESSMENT", 0)

    if assessment_type == "quiz":
        if attempts >= 2:
            queries.append(
                "The student has failed the quiz 2 or more times. Suggest coaching on overcoming quiz struggles."
            )
        if raw_score < 60:
            queries.append(
                f"The student scored {raw_score}% on the quiz, which is below 60%. Recommend ways to improve comprehension."
            )
    elif assessment_type == "exam":
        if attempts >= 1:
            queries.append(
                "The student has failed the exam at least once. Suggest coaching on mastering exam topics."
            )
        if raw_score < 60:
            queries.append(
                f"The student scored {raw_score}% on the exam, which is below 60%. Recommend exam preparation techniques."
            )

    if mastery_time < 120:
        queries.append(
            f"The student completed the course in {mastery_time} minutes, which is under 2 hours. Indicate potential skimming behavior."
        )

    if mastery_time < 30:
        queries.append(
            f"The student completed a lesson in {mastery_time} minutes, which is under 30 minutes. Suggest better engagement strategies."
        )

    return queries


def query_llm(user_query: str):
    response = process_user_query(user_query, conversation_history=[])
    return response
