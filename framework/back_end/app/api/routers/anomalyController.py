"""
Two end-points (both JWT-protected) are gonna be defined:
    1. POST /predict - image upload --> job_id
"""
import asyncio
from typing import List

from pydantic import BaseModel
from app.core.redis import redis_client
from app.core.alerts import add_alert, get_alerts, reset_alerts, Alert

from fastapi import APIRouter, Query, UploadFile, File, BackgroundTasks, Depends
from uuid import uuid4
import base64, json

from app.api.deps import get_current_user
from app.db.models import User

condition_variable = asyncio.Condition()


router = APIRouter(tags=["anomaly"])

"""dummy job into Redis"""
import redis
import base64
import uuid
import json

@router.post("/predict")
async def predict_image(
    # tells FastAPI to expect an image in multipart/form-data
    file: UploadFile = File(...), # kinda byte stream
    current_user: User = Depends(get_current_user)
):
    # read file content
    image_bytes = await file.read()
    # unique job identifier
    job_id = str(uuid4()) # globally unique id

    # asynchronously job queue - good for real-time processing
    job_data = {
        "id": job_id,
        "user": current_user.username,
        "image": base64.b64encode(image_bytes).decode('utf-8')  # encode image to base64
    }

    """
    https://medium.com/@omsaikrishnamadhav.lella/building-real-time-ml-inference-pipelines-with-fastapi-and-redis-073a6b8d5b01
    """
    # push a job to a queue (simulated here)
    # true real-time systems, decoupling via a job queue is preferred
    await redis_client.lpush("jobs", json.dumps(job_data)) # job queue simulated - json format is stored

    # high responsiveness : immediate return block
    # rather than running the model inline
    return {
        "job_id": job_id,
        "status": "queued"}

class RaiseAlertBody(BaseModel):
    job_id: str
    user: str
    label: str
    timestamp: str = None  

# GET /anomaly/since
@router.get("/since", response_model=List[Alert])
async def alerts_since():
    _list = await get_alerts()
    print(f"Alerts since: {_list}")
    return _list

# POST /anomaly/reset
@router.post("/reset")
async def reset_alert():
    async with condition_variable:   
        await reset_alerts()
        condition_variable.notify()
    return {"success": True, "message": "Alerts resetted successfully."}
