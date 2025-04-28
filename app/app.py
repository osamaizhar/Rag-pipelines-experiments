import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI
from dotenv import load_dotenv

from database.connections import check_database_connection
from routes import on_demand

load_dotenv()

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    check_database_connection()


app.include_router(on_demand.router)
