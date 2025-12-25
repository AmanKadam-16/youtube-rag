from fastapi.responses import JSONResponse
from fastapi import FastAPI
from src.yt_rag.router.api_router import router as master_router

app = FastAPI(
    title="Youtube RAG",
    description="""Capabilities include Question / Answering over youtube video | Transcript and even Visual questioning""",
)
app.include_router(master_router)


@app.get("/")
def root():
    return JSONResponse(status_code=200, content="System Operational.")
