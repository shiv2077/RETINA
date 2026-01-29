
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import httpx
from app.core.security import decode_token
from app.db.models import User

from sqlalchemy import select
from app.db.session import async_session


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    payload = decode_token(token)
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    async with async_session() as session:
        user = await session.scalar(select(User).where(User.username == username))
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    