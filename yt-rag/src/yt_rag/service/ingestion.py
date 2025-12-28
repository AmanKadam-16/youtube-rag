from src.yt_rag.utils.util import (
    transcript_extractor,
    chunk_transcript,
    create_embeddings,
)
from src.yt_rag.respository.ingestion import IngestionRepository
from src.yt_rag.schema import ingest as ingest_schema
from sqlalchemy.orm import Session
from typing import Any
from src.yt_rag.llm.tool.tool_function import rag_search


def fetch_video_transcript(video_url: str) -> list[ingest_schema.TranscriptExtract]:
    print("Step 01")
    util_response = transcript_extractor(video_url=video_url)
    return util_response


def process_embeddings(
    embeddings: list[Any],
    chunks_list: list[str],
    collection_code: str,
    video_url: str,
    timestamps: list[float],
    db: Session,
) -> ingest_schema.EmbeddingStatus:
    embeddings_entries = []
    for idx, embedding in enumerate(embeddings):
        chunk_metadata = ingest_schema.ChunkMetadata(
            chunk_no=idx, yt_video_url=video_url, start_timestamp=timestamps[idx]
        )
        store_entry = ingest_schema.EmbeddingEntry(
            chunk=chunks_list[idx],
            collection_code=collection_code,
            embedding=embedding.embedding,
            chunk_metadata=chunk_metadata.model_dump_json(),
        )
        embeddings_entries.append(store_entry)
    ingestion_result = IngestionRepository.add_embedding(
        embedding_entries=embeddings_entries, db=db
    )
    return ingestion_result


def process_transcript_chunks(
    chunks_list: list[str],
    collection_code: str,
    video_url: str,
    timestamps: list[float],
    db: Session,
) -> ingest_schema.EmbeddingStatus:
    print("Step 03")
    chunk_embeddings = create_embeddings(chunks_list=chunks_list)
    processed_embeddings_result = process_embeddings(
        embeddings=chunk_embeddings,
        chunks_list=chunks_list,
        collection_code=collection_code,
        video_url=video_url,
        timestamps=timestamps,
        db=db,
    )
    return processed_embeddings_result


def ingest_yt_video(
    video_info: ingest_schema.VideoIngest, db: Session
) -> ingest_schema.EmbeddingStatus:
    print("Transcript Extraction Started.")
    transcript = fetch_video_transcript(video_url=video_info.yt_video_url)
    transcript_chunks, timestamps = chunk_transcript(transcript_info=transcript)
    ingestion_result = process_transcript_chunks(
        chunks_list=transcript_chunks,
        collection_code=video_info.collection_code,
        video_url=video_info.yt_video_url,
        timestamps=timestamps,
        db=db,
    )
    return ingestion_result


def get_context(user_query: str, db: Session):
    result = rag_search({"user_query": user_query})
    return result


def get_video_frame(): ...


"""
    @staticmethod
    def list_collections(db: Session):
        collections = (
            db.query(EmbeddingStore)
            .distinct(EmbeddingStore.collection_code)
            .all()
        )
        return collections
"""


def list_collections(db: Session) -> list[str]:
    collections = IngestionRepository.list_collections(db=db)
    collection_list = [col.collection_code for col in collections]
    return collection_list
