# app/integrations/active_learning.py
import os
import httpx
from typing import List, Dict

AL_BASE = os.getenv("AL_BASE_URL", "http://127.0.0.1:9000")

async def al_select(limit: int = 10) -> List[str]:
    url = f"{AL_BASE}/active-learning/select"
    params = {"limit": limit} # by Query method of FastAPI
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()["selected"]

