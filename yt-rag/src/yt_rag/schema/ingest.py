from pydantic import BaseModel, Field
from typing import Any


class VideoIngest(BaseModel):
    yt_video_url: str = Field(
        ...,
        pattern="^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(?:-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|live\/|v\/)?)([\w\-]+)(\S+)?$",
    )
    collection_code: str = Field(..., min_length=3)


class TranscriptExtract(BaseModel):
    text: str
    start: float
    duration: float


class EmbeddingEntry(BaseModel):
    chunk: str = Field(..., min_length=1)
    embedding: list[float]
    collection_code: str = Field(..., min_length=3)
    chunk_metadata: Any


class EmbeddingStatus(BaseModel):
    total_entries: int


class ChunkMetadata(BaseModel):
    chunk_no: int
