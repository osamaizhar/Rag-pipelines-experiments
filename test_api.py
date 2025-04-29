from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Define a simple app without database dependencies
app = FastAPI()

# Define the request model
class OnDemandReqBody(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    user_query: str

# Define a simple response model
class StandardResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[dict] = None

@app.post("/process", response_model=StandardResponse)
async def process_query(request: OnDemandReqBody):
    try:
        # Validate request
        if not request.user_query or not isinstance(request.user_query, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User query is required and must be a string."
            )
        
        # Simple mock response
        return StandardResponse(
            status_code=status.HTTP_200_OK,
            message="Success",
            data={"session_id": request.session_id or "test-session", 
                  "response": f"Mock response to: {request.user_query}"}
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )

# Run the app directly when this file is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7012)