from logging import log
import os
import uuid

# Endpoint to submit expert label
from fastapi import Body, HTTPException, Query
# Endpoint to fetch anomaly records pending review
from fastapi import APIRouter, Depends

from sqlalchemy import select
from app.db.models import AnomalyRecord
from app.db.session import async_session

from app.api.routers.learningService import batch_fetch_pending_reviews, update_expert_feedback
from app.activeLearningClient import al_select

router = APIRouter(tags=["learning"])

@router.get("/review")
async def get_pending_reviews(limit: int = Query(10, ge=1, le=15)) -> dict:
    """
    Fetch a batch of anomaly records pending expert review, 
    picked by the active learning black box
    """
    print(f"Fetching {limit} pending reviews from active learning service...")
    try:
        selected_ids = await al_select(limit=limit)  # e.g. returns ["id1","id2"] or []
    except Exception as e:
        # propagate as 502 so the frontend sees a clean error instead of 500
        log.exception("Active-learning select failed")
        raise HTTPException(status_code=502, detail=f"Active-learning service failed: {e}")
    
    if not selected_ids:
        log.info("Active-learning returned 0 IDs.")
        return {"Records": []}
    
    # fetch full records from DB
    records = await batch_fetch_pending_reviews(selected_ids)
    
    return {"Records": [record.to_dict() for record in records]}
"""
The {image_id} inside "/label/{image_id}" is a path parameter.
When a request hits /label/12345, 
FastAPI will parse 12345 from the URL 
and pass it into the function as the image_id arg
"""

# POST: Submit expert label only (e.g. confirming anomaly or not)
@router.post("/label/{image_id}")
async def submit_expert_label(image_id: str, label: str = Body(..., embed=True)) -> dict:
    """Submit expert label for a specific anomaly record."""
    expert_label = True if label.lower() == "anomalous" else False
    result = await update_expert_feedback(image_id, expert_label)
    return result

# Endpoint to submit expert label with classification
@router.post("/label/{image_id}/class")
async def submit_classification_label(image_id: str, classification: str = Body(..., embed=True)) -> dict:
    """Submit expert label and classification for a specific anomaly record."""
    expert_label = True # taken for granted
    result = await update_expert_feedback(image_id, expert_label, final_classification=classification)
    return result
