import asyncio

from typing import Optional, List
from sqlalchemy import select

from app.db.models import AnomalyRecord, User
from app.db.session import async_session

from app.schemas.payloads import payloads

# POST: Submit expert label only (e.g. confirming anomaly or not)
async def update_expert_feedback(
    image_id: str,
    expert_label: bool,
    final_classification: Optional[str] = None
):
    """Update a record with expert feedback."""
    # Use it across services, routers, and background jobs
    # Avoid repeating DB connection config in every file
    async with async_session() as session:
        result = await session.execute(
            select(AnomalyRecord).where(AnomalyRecord.id == image_id)
        )
        record = result.scalar_one_or_none()        
        if not record:
            raise ValueError(f"Record with id {image_id} not found.")

        record.reviewed = True
        record.expert_label = "anomalous" if expert_label else "normal"
        record.final_classification = final_classification
        print("Relabeled record:", record.expert_label, record.final_classification)
        await session.commit()
        return {"success": True, "message": "Record updated successfully."}


async def batch_fetch_pending_reviews(selected_ids: List[str]) -> List[AnomalyRecord]:
    """Fetch a batch of anomaly records pending expert review."""
    async with async_session() as session:
        result = await session.execute(
            select(AnomalyRecord)
            .where(AnomalyRecord.id.in_(selected_ids))
        )   
        records = result.scalars().all() # already a list of AnomalyRecord objects
        return records

async def flush_buffer_to_active_learning(results: List[dict]) -> bool:
    """Flush buffer to active learning if threshold is reached."""
    if not results:
        print("No results to flush to active learning.")
        return False
    
    for result in results:
        print(f"Processing result for image_id: {result['image_id']}")
        username = result["username"]
        image_id = result["image_id"]
        payload_record = payloads[image_id]

        """
        if anomalous, return True
        otherwise False
        """
        payload_record["unsupervised_label"] = result["is_anomalous"]
        # if supervised models are completely done on this image
        while payload_record["supervised_label"] is None and payload_record["final_classification"] is None:
            # Wait for supervised result to be set
            print(f"Waiting for supervised results for image_id: {image_id}")
            await asyncio.sleep(2)

        # save record to database
        async with async_session() as session:
            print(f"Saving record for image_id: {image_id} to database")
            record = AnomalyRecord(
                id=image_id,
                user=username,
                file_path=payload_record["file_path"],
                timestamp=payload_record["timestamp"],
                unsupervised_label=bool(payload_record["unsupervised_label"]),
                supervised_label=bool(payload_record["supervised_label"]),
                mismatch=bool(payload_record["mismatch"]),
                reviewed=False,
                expert_label=None,
                final_classification=payload_record["final_classification"]
            )
            session.add(record)
            await session.commit()
        # remove from payloads after processing
        payloads.pop(image_id, None)
    print("All results flushed to active learning.")
    return True
