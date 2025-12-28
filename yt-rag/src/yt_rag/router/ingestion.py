from fastapi.routing import APIRouter
from src.yt_rag.schema import ingest as ingest_schema
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi import Depends, HTTPException
from src.yt_rag.database.db import get_db
from src.yt_rag.service import ingestion as ingestion_service

# from src.yt_rag.schema.ingest import TranscriptExtract
# from src.yt_rag.utils.test import generate_frames_and_save

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("")
def ingest_video(video_info: ingest_schema.VideoIngest, db: Session = Depends(get_db)):
    try:
        service_response = ingestion_service.ingest_yt_video(
            video_info=video_info, db=db
        )
        return service_response
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code, content=f"HTTP Error : {he.detail}"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=f"Server Error : {e}")


@router.get("/collections")
def get_collections(db: Session = Depends(get_db)) -> list[str]:
    try:

        service_response = ingestion_service.list_collections(db=db)
        return service_response
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code, content=f"HTTP Error : {he.detail}"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content=f"Server Error : {e}")


# @router.get("/get-video-transcript", response_model=list[TranscriptExtract])
# def get_transcript(yt_video_url: str):
#     try:
#         service_response = ingestion_service.fetch_video_transcript(yt_video_url)
#         return service_response
#     except HTTPException as he:
#         return JSONResponse(
#             status_code=he.status_code, content=f"HTTP Error : {he.detail}"
#         )
#     except Exception as e:
#         return JSONResponse(status_code=500, content=f"Server Error : {e}")


# @router.post("/get-context")
# def get_context(user_query: str, db: Session = Depends(get_db)):
#     try:
#         service_response = ingestion_service.get_context(user_query=user_query, db=db)
#         return service_response
#     except HTTPException as he:
#         return JSONResponse(
#             status_code=he.status_code, content=f"HTTP Error : {he.detail}"
#         )
#     except Exception as e:
#         return JSONResponse(status_code=500, content=f"Server Error : {e}")


# @router.get("/trigger-frame-generation")
# def frame_generation():
#     try:
#         service_response = generate_frames_and_save()
#         return service_response
#     except HTTPException as he:
#         return JSONResponse(
#             status_code=he.status_code, content=f"HTTP Error : {he.detail}"
#         )
#     except Exception as e:
#         return JSONResponse(status_code=500, content=f"Server Error : {e}")
