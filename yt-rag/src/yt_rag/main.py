from fastapi.responses import JSONResponse
from fastapi import FastAPI
from src.yt_rag.router.api_router import router as master_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Youtube RAG",
    description="""Capabilities include Question / Answering over youtube video | Transcript and even Visual questioning""",
)
app.include_router(master_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return JSONResponse(status_code=200, content="System Operational.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)