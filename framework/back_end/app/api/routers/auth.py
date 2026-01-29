"""
OAuth2 Password Flow - FastAPI provides a dependency to extract the bearer token from requests.
/token endpoint is to be used for clients to get a JWT by a valid username and password.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Body

from app.core.security import verify_password, create_access_token, hash_password
from app.db import session
from app.db.session import async_session
from app.db.models import User

from sqlalchemy import select 
from sqlalchemy.exc import IntegrityError

from app.schemas.user import Token

router = APIRouter(tags=["auth"])

# FastAPI is told where to find the token
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@router.post("/token", response_model=Token)
# the response will be validated based on a Pydantic token
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2PasswordRequestForm is a FastAPI dependency
    # autamatically extracts username and password from the request body
    # Depends() tells FastAPI: “Inject this form into the function when a request comes in”
        # it automatically calls the function you pass in 
        # injects its return value into your endpoint logic
    print("Login attempt for user:", form_data.username)
    async with async_session() as session:
        stmt = select(User).where(User.username == form_data.username)
        result = await session.scalar(stmt)

        # find user by username 
        if not result or not verify_password(form_data.password, result.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )

        access_token = create_access_token(data={"sub": result.username})
        # "sub" is a standard claim for the subject of the token, usually the user
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

"""
  "access_token": "<JWT_string>",
  "token_type": "bearer"
"""

@router.post("/register")
async def register(
    username: str = Body(...),
    password: str = Body(...)
):
    async with async_session() as session:
        stmt = select(User).where(User.username == username)
        existing_user = await session.scalar(stmt)

        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )
        # by app.core.security.create_access_token
        hashed_password = hash_password(password)
        new_user = User(username=username, hashed_password=hashed_password)
        session.add(new_user)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(
                status_code=500,
                detail="User creation failed"
            )
        return {"message": "User created successfully"}