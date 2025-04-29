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
    # Modified to handle database connection failure
    try:
        db_available = check_database_connection()
        app.state.db_available = db_available
    except Exception as e:
        print(f"Error checking database: {e}")
        app.state.db_available = False
    print(f"Application starting with database available: {app.state.db_available}")


app.include_router(on_demand.router)
