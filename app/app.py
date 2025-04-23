from fastapi import FastAPI
from dotenv import load_dotenv
from app.database.connections import check_database_connection
from app.routes import query_llm_routes

load_dotenv()

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    check_database_connection()

app.include_router(query_llm_routes.router)
