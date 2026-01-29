# app/worker.py
import asyncio

from app.core.redis import redis_client  # already works
from app.worker_loop import worker_loop  # <- this should contain your async def worker_loop()

if __name__ == "__main__":
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        print("Worker interrupted; shutting down gracefully.")