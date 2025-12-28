from sqlalchemy.orm import Session
from sqlalchemy import asc
from src.yt_rag.model.model import EmbeddingStore
from src.yt_rag.schema.ingest import EmbeddingEntry, EmbeddingStatus


class IngestionRepository:

    @staticmethod
    def add_embedding(
        embedding_entries: list[EmbeddingEntry], db: Session
    ) -> EmbeddingStatus:
        print("Len of embedding entries -- >> ", len(embedding_entries))
        for embedding_entry in embedding_entries:
            new_embedding = EmbeddingStore(
                chunk=embedding_entry.chunk,
                collection_code=embedding_entry.collection_code,
                embedding=embedding_entry.embedding,
                chunk_metadata=embedding_entry.chunk_metadata,
            )
            db.add(new_embedding)
            db.commit()
        total_embeddings = db.query(EmbeddingStore).count()
        return EmbeddingStatus(total_entries=total_embeddings)

    @staticmethod
    def collection_exists(collection_code: str, db: Session):
        collection_exists = (
            db.query(EmbeddingStore)
            .filter(
                EmbeddingStore.collection_code == collection_code,
                EmbeddingStore.is_active,
            )
            .first()
        )
        return collection_exists

    @staticmethod
    def get_similar_chunks(query_embedding: list[float], db: Session, top_k=10, collection_id:str = ""):
        distance = EmbeddingStore.embedding.cosine_distance(query_embedding)
        similar_chunks = (
            db.query(EmbeddingStore, distance.label("distance"))
            .filter(EmbeddingStore.is_active, EmbeddingStore.collection_code == collection_id)
            .order_by(asc(distance))
            .limit(top_k)
            .all()
        )
        resultant_chunks = []
        for chunk, cos_dist in similar_chunks:
            sim_score = 1 - cos_dist
            chunk_info = {
                "text": chunk.chunk,
                "metadata": chunk.chunk_metadata,
                "similarity_score": sim_score,
            }
            resultant_chunks.append(chunk_info)
        return resultant_chunks
    
    @staticmethod
    def list_collections(db: Session):
        collections = (
            db.query(EmbeddingStore)
            .distinct(EmbeddingStore.collection_code)
            .all()
        )
        return collections
