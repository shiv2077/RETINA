import asyncio
from app.db.session import async_session
from app.db.models import User
from sqlalchemy import select

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def create_user():
    async with async_session() as session:
        result = await session.scalar(select(User).where(User.username == "test_user"))
        if result:
            print("User already exists.")
            return
        
        user = User(username="test_user", hashed_password=pwd_context.hash("password123"))
        session.add(user)
        await session.commit()
        print("User created.")

asyncio.run(create_user())
