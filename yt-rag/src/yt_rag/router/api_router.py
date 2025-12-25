from fastapi.routing import APIRouter
from src.yt_rag.router.chatbot import router as chat_router
from src.yt_rag.router.ingestion import router as ingestion_router


router = APIRouter()
router.include_router(chat_router)
router.include_router(ingestion_router)
