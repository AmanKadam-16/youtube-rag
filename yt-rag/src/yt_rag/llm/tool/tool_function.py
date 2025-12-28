from src.yt_rag.utils.util import create_query_embedding
from src.yt_rag.respository.ingestion import IngestionRepository
from src.yt_rag.database.db import get_db_session


def rag_search(arg: dict, collection_id:str):
    with get_db_session() as db:
        user_query = arg["user_query"]
        embedding = create_query_embedding([user_query])

        return IngestionRepository.get_similar_chunks(embedding, db=db, top_k=10, collection_id=collection_id)
