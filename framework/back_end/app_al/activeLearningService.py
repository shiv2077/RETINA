# active learning service 
import os, random, asyncio
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Query
from pydantic import BaseModel

from sqlalchemy import String, Boolean, DateTime, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# ---------- Config ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://mv4qc_user:mv4qc123@127.0.0.1:5432/mv4qc_db")

# mimic “prep” time (seconds) to simulate slow selection
DEFAULT_PREP_SEC = int(os.getenv("AL_PREP_SECONDS", "2"))

# ---------- ORM mirror (only what we need) ----------
class Base(DeclarativeBase): 
    pass

class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_path: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    unsupervised_label: Mapped[bool] = mapped_column(Boolean)
    supervised_label: Mapped[bool] = mapped_column(Boolean)
    mismatch: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    expert_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    final_classification: Mapped[Optional[str]] = mapped_column(String, nullable=True)

import logging, os
logging.getLogger("uvicorn.error").info(f"AL DATABASE_URL: {os.getenv('DATABASE_URL')}")

engine = create_async_engine(
    DATABASE_URL,
    echo=True,                  # helpful while debugging
    pool_pre_ping=True,         # validate connections before use (fixes stale conns)
    pool_recycle=1800,          # recycle connections periodically
    connect_args={"ssl": False} # same as your main app
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# ---------- API ----------
app = FastAPI(title="Active Learning Service")

@app.get("/active-learning/select")
async def select_for_review(
    limit: int = Query(10, ge=1, le=15),  # upper limit to return
):
    """
    Return up to `limit` image_ids that are not yet reviewed.
    May return fewer (dropout). Simulates prep time.
    """

    # Fixed knobs (you can turn these into Query params later if you want)
    prep_seconds = DEFAULT_PREP_SEC          # artificial delay
    lookback_minutes = 60 * 24 * 14          # 14-day window
    dropout_rate = 0.25                      # may return fewer than requested

    await asyncio.sleep(prep_seconds)

    since = datetime.utcnow() - timedelta(minutes=lookback_minutes)
    async with SessionLocal() as session:
        stmt = (
            select(AnomalyRecord.id)                  # only fetch IDs
            .where(AnomalyRecord.reviewed == False)
            .where(AnomalyRecord.timestamp >= since)
            .order_by(AnomalyRecord.timestamp.desc())
        )
        rows : List[str] = (await session.execute(stmt)).scalars().all()

    if not rows:
        return {"selected": []}

    pool = len(rows)                     # how many available candidates
    available = min(limit, pool)        # cap by requested limit

    # Lower bound based on dropout; upper bound is all available.
    # Now choose a random keep in [lower_bound, available], inclusive.
    lower_bound = max(0, int(round(available * (1.0 - dropout_rate))))
    upper_bound = available
    if upper_bound < lower_bound:
        lower_bound = upper_bound  # safety (can happen with rounding edge cases)

    keep = random.randint(lower_bound, upper_bound) if upper_bound > 0 else 0

    selected = random.sample(rows, keep) if keep > 0 else []

    return {"selected": selected}
