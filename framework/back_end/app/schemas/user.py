# A Pydantic model for user data
"""
https://medium.com/@suganthi2496/fastapi-security-essentials-using-oauth2-and-jwt-for-authentication-7e007d9d473c
"""
from pydantic import BaseModel

# in a real system db use for data storage - users, images etc.
# here is used a in-memory store
"""
as suggested in the article, a fake_users_db dict mapping usernames to user data can be maintained
https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
it is preferable for demonstration purposes / not production use
"""
# pydantic.BaseModel provides input validation, API docs, easy type enforcement
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase): # registration
    password: str  # plain-text password for signup (will be hashed)

class User(UserBase): # login or lookup
    id: int
    # Note: in a real app, you wouldn't expose hashed_password in response

class Token(BaseModel): # login response
    access_token: str
    token_type: str

class TokenData(BaseModel): # username in token
    username: str | None = None
