from fastapi import Depends, FastAPI
import uvicorn
from app.api.deps import get_current_user
from app.api.routers import anomalyController, auth, learningController
from app.core.redis import redis_client

import asyncio

app = FastAPI()

"""

CORS = Cross-Origin Resource Sharing
Browsers block requests from one “origin” to another 
unless the server explicitly says it’s okay.
origin = "scheme://hostname:port"

Even though both are localhosts port numbers are different,
so the browser can see them different origins by CORS
"""


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# registering routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
# get_current_user as a dependency 
# for all anomaly routes at once 
# ensuring every route in that router is protected
app.include_router(anomalyController.router, prefix="/anomaly", dependencies=[Depends(get_current_user)])

app.include_router(learningController.router, prefix="/learning", dependencies=[Depends(get_current_user)])



@app.on_event("startup")
async def startup_event():
    # FastAPI app starts, it sends a ping to Redis
    try:
        await redis_client.ping()
        print("Connected to Redis")
    except Exception as e:
        print("Redis connection failed:", e)
    
    # await init_db()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)