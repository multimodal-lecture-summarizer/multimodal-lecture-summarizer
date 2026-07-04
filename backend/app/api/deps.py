import uuid
from datetime import datetime, timedelta
from typing import Generator, Optional
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AuthException, ForbiddenException, NotFoundException
from app.core.constants import UserRole
from app.models.user import User

import bcrypt

# OAuth2 scheme config (pointing to token login endpoint)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password matches its bcrypt hash."""
    try:
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash for a plaintext password."""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def create_access_token(
    subject: str, role: UserRole, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Generates a JWT access token for a user.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {"exp": expire, "sub": str(subject), "role": role.value}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm="HS256"
    )
    return encoded_jwt


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency to validate JWT token and return the current authenticated User.
    Throws AuthException if token is invalid or user doesn't exist.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise AuthException(message="Could not validate credentials")
        user_id = uuid.UUID(user_id_str)
    except (JWTError, ValueError):
        raise AuthException(message="Could not validate credentials")

    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise AuthException(message="User not found or credentials invalid")
    if not user.is_active:
        raise AuthException(message="Inactive user account")

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency helper ensuring user is active."""
    return current_user


def check_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to verify the current user is an Administrator.
    Throws ForbiddenException if user is not an admin.
    """
    if current_user.role != UserRole.ADMIN:
        raise ForbiddenException(message="The user does not have enough privileges")
    return current_user
