import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pydantic import BaseModel
from app.core.redis import redis_client

ALERTS_KEY = "alerts"   # Redis list name

class Alert(BaseModel):
    job_id: str
    user: str
    label: str
    timestamp: str


async def add_alert(job_id: str, user: str, label: str):
    """
    Add an alert to the alerts list
    """
    alert = Alert(
        job_id=job_id,
        user=user, 
        label=label,
        timestamp=datetime.utcnow().isoformat()
    )
    # push serialized JSON to Redis list (LPUSH = newest first)
    await redis_client.lpush(ALERTS_KEY, json.dumps(alert.model_dump()))
    return alert.model_dump()

async def get_alerts() -> List[Alert]:
    """
    Get the current alerts list
    """
    items = await redis_client.lrange(ALERTS_KEY, 0, -1)
    results: List[Alert] = []
    for item in items:
        try:
            data = json.loads(item)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {item}")
            continue
        try:
            results.append(Alert.model_validate(data))
        except Exception as e:
            print(f"Failed to validate Alert model: {e}, data: {data}")
    return results

async def reset_alerts():
    """
    Reset the alerts count and clear the list
    """
    await redis_client.delete(ALERTS_KEY)
