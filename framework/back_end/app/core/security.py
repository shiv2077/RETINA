from passlib.context import CryptContext
# CryptContext is a manager that handles password hashing.

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # bcrypt is the algorithm

def hash_password(password: str) -> str: # sign up
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool: # login
    return pwd_context.verify(plain_password, hashed_password)

# simulated in-memory db unit

"""
Generate a JWT access token
"""
import jwt
from datetime import datetime, timedelta
from app.core.config import settings

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(data: dict) -> str:
    # with an expiry time, a JWT access token is generated
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire}) # expiration added to the token
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

"""
Decode and validate a JWT token. Raise error if invalid or expired.
"""

from fastapi import HTTPException, status
from jwt import PyJWTError

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # if decode succeeds, you get the data back

        return payload
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )