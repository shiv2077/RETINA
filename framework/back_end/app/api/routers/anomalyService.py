import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from datetime import datetime
from typing import Optional

from PIL import Image
import io

import numpy as np

from app.api.deps import get_current_user
from app.db.models import AnomalyRecord
from app.db.session import async_session
from app.schemas.user import User

from app.core.config import MEDIA_ROOT, ANOMALY_ROOT

from app.api.routers.anomalyController import condition_variable
from app.core.alerts import add_alert, get_alerts, reset_alerts, Alert


async def raise_alert(
    job_id: str, 
    user: str,
    _class: str
):
    async with condition_variable:   
        await add_alert(job_id, user, _class)
        condition_variable.notify()
    return {"success": True, "message": "Alert raised successfully."}

async def save_anomaly_to_disk_and_db(
    unsupervised_result: dict,
    supervised_result: dict,
    mismatch: bool,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    save the image to file system and log record to database.
    this method has supervised and unsupervised anomaly results as parameters.
    """
    image_bytes = await file.read()
    # Create unique ID and file path
    image_id = str(uuid.uuid4()) # str(uuid4()) okay too
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_id}_{timestamp}.png"
    user_dir = os.path.join(ANOMALY_ROOT, current_user.username)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, filename)

    # Save image to disk
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save(filepath)

    # Save metadata to database
    async with async_session() as session:
        record = AnomalyRecord(
            id=image_id,
            user=current_user,
            file_path=filepath,
            timestamp=datetime.utcnow(),
            unsupervised_label=unsupervised_result["is_anomalous"],
            supervised_label=supervised_result["is_anomalous"],
            mismatch= mismatch,
            reviewed=False,
            expert_label=None,
            final_classification=None
        )
        session.add(record)
        await session.commit()

    return image_id, filepath

class DummyInferencer:
    def predict(self, image: Image.Image):
        # dummy result structure mimicking real TorchInferencer output
        return type("Prediction", (), {
            "pred_score": np.random.rand(),
            "pred_label": "anomaly"
        })()

