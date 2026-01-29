import redis.asyncio as redis
# redis.asyncio is the async-compatible client 
from app.core.config import settings

# create Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True  # ensures we get strings instead of bytes
    # Redis usually returns bytes
    # but we want str (.get() or .lpush())
)
