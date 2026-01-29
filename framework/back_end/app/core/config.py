"""
the referenced article is:
https://medium.com/@suganthi2496/fastapi-security-essentials-using-oauth2-and-jwt-for-authentication-7e007d9d473c
"""

import os
from pydantic_settings import BaseSettings
from dotenv import find_dotenv
# load the values mentioned in .env file
"""
Defines a class Settings that inherits from BaseSettings, 
making it capable of loading configuration from external sources.
"""
class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256" #  a common algorithm used for JWTs
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REDIS_URL: str = "redis://localhost:6379"

    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: str

        # Uvicorn / logging (these are the ones causing your error)
    UVICORN_HOST: str = "127.0.0.1"
    UVICORN_PORT: int = 8000
    UVICORN_WORKERS: int = 1
    LOG_LEVEL: str = "info"

    class Config:
        env_file = find_dotenv(".env")  # load variables from .env

# global settings objecr 
# that can be imported anywhere in the app
settings = Settings()

"""
config for the paths of the file system that the images will be saved
"""
# base directory of your project (where main.py or main folder lives)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDIA_ROOT = os.path.join(BASE_DIR, "..", "..", "media")
ANOMALY_ROOT = os.path.join(MEDIA_ROOT, "anomalies")