import uuid
from datetime import datetime
from typing import Optional
from pydantic import EmailStr, Field
from app.schemas.base import CamelModel
from app.core.constants import UserRole


class UserBase(CamelModel):
    email: EmailStr = Field(
        ..., description="The unique email address of the user"
    )


class UserCreate(UserBase):
    password: str = Field(
        ..., min_length=6, description="The plaintext password of the user"
    )


class UserLogin(CamelModel):
    email: EmailStr = Field(..., description="The email address of the user")
    password: str = Field(..., description="The user's password")


class UserDTO(UserBase):
    user_id: uuid.UUID = Field(
        ..., description="The unique UUID of the registered user"
    )
    role: UserRole = Field(
        UserRole.USER, description="The system role assigned to the user"
    )
    created_at: datetime = Field(
        ..., description="The date and time the user account was created"
    )
    last_login: Optional[datetime] = Field(
        None, description="The date and time of the last login session"
    )
    is_active: bool = Field(
        True, description="Indicates whether the account is currently active"
    )


class Token(CamelModel):
    access_token: str = Field(
        ..., description="The OAuth2 JWT bearer access token string"
    )
    token_type: str = Field(
        "bearer", description="The token type, which is typically 'bearer'"
    )


class TokenData(CamelModel):
    user_id: Optional[uuid.UUID] = Field(
        None, description="The user UUID parsed from the token subject"
    )
    role: Optional[UserRole] = Field(
        None, description="The role claims parsed from the token"
    )
