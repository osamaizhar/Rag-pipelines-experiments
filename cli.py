import typer
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

cli = typer.Typer()

PORT = int(os.getenv("PORT", 7010))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", 7020))

@cli.command()
def start():
    """Launch the backend service."""
    uvicorn.run("app.app:app", host="0.0.0.0", port=PORT)
    
@cli.command()
def dev():
    """Launch the backend service."""
    uvicorn.run("app.app:app", host="0.0.0.0", port=PORT, reload=True)

@cli.command()
def gradio():
    """Launch the Gradio chatbot interface."""
    uvicorn.run("app.gradio-app:app", host="0.0.0.0", port=GRADIO_PORT, reload=True)

if __name__ == "__main__":
    cli()
