from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import Generator, Iterator
from sqlalchemy import create_engine
from contextlib import contextmanager
from src.yt_rag.core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(DATABASE_URL)

session_local = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


def get_db() -> Generator:
    db = session_local()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Iterator[Session]:
    db = session_local()
    try:
        yield db
    finally:
        db.close()
