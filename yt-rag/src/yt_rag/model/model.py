from sqlalchemy import (
    Column,
    Text,
    UUID,
    JSON,
    Boolean,
    String,
    DateTime,
    func,
)
import uuid
from src.yt_rag.database.db import Base
from pgvector.sqlalchemy.vector import VECTOR
from src.yt_rag.core.config import settings


# Columns Name - id, chunk, embedding, metadata
class EmbeddingStore(Base):
    __tablename__ = "embedding_store"
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    chunk = Column(Text, nullable=False)
    embedding = Column(VECTOR(settings.EMBEDDING_DIMENSION), nullable=False)
    collection_code = Column(String, nullable=False)
    chunk_metadata = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )
