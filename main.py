from fastapi import FastAPI, Request
from pydantic import BaseModel
# import sys
# import os

# # Add the parent directory to sys.path
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)


from archive.query_llm import process_user_query



# Initialize the FastAPI app
app = FastAPI()

# Define the request body
class UserQuery(BaseModel):
    user_query: str

@app.post("/process")
async def process_query(query: UserQuery):
    user_query = query.user_query

    # Call the function from groq_test.ipynb
    response = process_user_query()
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
