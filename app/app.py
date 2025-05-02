import sys
import os

from fastapi.exceptions import RequestValidationError, HTTPException

from app.utils.exceptions.handlers import (
    http_exception_handler,
    validation_exception_handler,
)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI
from dotenv import load_dotenv

from database.connections import check_database_connection
from routes import on_demand, failure_driven

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-session-id"],
)


@app.on_event("startup")
async def on_startup() -> None:
    check_database_connection()


app.include_router(on_demand.router)
app.include_router(failure_driven.router)
