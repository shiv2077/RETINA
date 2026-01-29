# app/worker.py
from datetime import datetime
import io
import asyncio, json, base64
import os

from PIL import Image
import httpx
from sqlalchemy import select

from app.schemas.payloads import payloads

from app.core.redis import redis_client
from app.core.buffer import should_run_unsupervised, reset_buffer, get_buffer_capacity, get_buffer_list
from app.core.alerts import add_alert, get_alerts, reset_alerts, Alert

from app.api.routers.unsupervisedAnomalyService import run_inference
from app.api.routers.supervisedAnomalyService import run_supervised_inference
from app.api.routers.supervisedClassificationService import run_classification_inference
from app.api.routers.learningService import flush_buffer_to_active_learning
from app.api.routers.anomalyService import raise_alert

from app.db.session import async_session
from app.db.models import AnomalyRecord  # assuming User model exists for DB lookup
from app.db.models import User  # assuming User schema exists for DB lookup

condition_variable = asyncio.Condition()

async def job_consumer_loop():
    print("Job consumer started...")
    while True:
        """
        even if process_job() takes time, 
        the loop goes back to waiting 
        for the next Redis job without blocking
        """
        job = await redis_client.brpop("jobs", timeout=0)
        _, data = job # _ takes Redis key name : can be discarded
        asyncio.create_task(process_job(data))  # non-blocking

async def unsupervised_checker_loop():
    print("Unsupervised checker started...")
    while True:
        await asyncio.sleep(5)  # check every few seconds
        buffer_data = get_buffer_list()
        print(f"unsupervised_checker_loop: size is {len(buffer_data)}")
        if buffer_data and len(buffer_data) >= get_buffer_capacity():
            async with condition_variable:
                print("Buffer is full, running unsupervised inference...")
                buffer_copy = dict(buffer_data)
                reset_buffer()

            results = []
            for key in buffer_copy.keys():
                try:
                    result = run_inference(key, buffer_copy[key][1])
                    print(f"Unsupervised inference result for {key}: {result}")
                    result["username"] = buffer_copy[key][0]
                    results.append(result)
                except Exception as e:
                    results.append(e)

            try:
                await flush_buffer_to_active_learning(results)
            except Exception as e:
                print(f"Error in flush_buffer_to_active_learning: {e}")

async def worker_loop():
    await asyncio.gather(
        job_consumer_loop(),
        unsupervised_checker_loop()
    )

async def process_job(data):
    job_data = json.loads(data)
    user = job_data["user"] # username
    job_id = job_data["id"]
    image_str = job_data["image"]
    image_bytes = base64.b64decode(image_str.encode("utf-8"))

    timestamp = datetime.utcnow()
    user_dir = f"images/{user}"
    os.makedirs(user_dir, exist_ok=True)
    filename = f"{job_id}_{timestamp:%Y%m%d_%H%M%S}.jpg"
    file_path = os.path.join(user_dir, filename)
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(file_path)

    payload_record = {
        "id": job_id,
        "user": None,
        "file_path": file_path,
        "timestamp": timestamp,
        "unsupervised_label": None,
        "supervised_label": None,
        "mismatch": None,
        "reviewed": False,
        "expert_label": None,
        "final_classification": None
    }
    payloads[job_id] = payload_record

    # Track image for buffer (won't block loop)
    async with condition_variable:
        print("unsupervised model buffer is incremented")
        should_run_unsupervised(job_id, user, image_bytes)
        condition_variable.notify()
        print("unsupervised model buffer is notified")

    # Run supervised model immediately
    supervised_result = run_supervised_inference(image_bytes)
    payload_record["supervised_label"] = supervised_result["is_anomalous"]
    print("supervised #1 model is just run, timestamp: ", timestamp)

    if supervised_result["is_anomalous"]:
        cls_result = run_classification_inference(image_bytes)
        payload_record["final_classification"] = cls_result["label"]
        print("supervised #2 model is just run")

        try:
            await add_alert(
                job_id=job_id,
                user=user,
                label=cls_result["label"]
            )
        except Exception as e:
            print(f"Error raising alert: {e}")

        try:
            alerts_now = await get_alerts()   # import from app.core.alerts
            print(f"************************* count: {len(alerts_now)}")
        except Exception as e:
            print(f"Failed to fetch alerts from Redis: {e}")


    else:
        payload_record["final_classification"] = "normal"
        print("supervised #2 model is not run, image is normal")



