# app/db/session.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Load from environment or hardcode (for local development)
DATABASE_URL = settings.DATABASE_URL
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://mv4qc_user:mv4qc123@127.0.0.1:5432/mv4qc_db")

# Async engine
engine = create_async_engine(
    DATABASE_URL, 
    echo=True,
    connect_args={"ssl": False})

# Session factory
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

"""
You use async_session and await session.execute(...)

You connect with a URL like:
postgresql+asyncpg://mv4qc_user:mv4qc123@localhost:5432/mv4qc_db
"""