"""Auth endpoint — login, register, JWT token management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.security import create_access_token

router = APIRouter()


@router.post("/login")
async def login(body: dict):
    """Authenticate user and return JWT token."""
    email = body.get("email")
    password = body.get("password")
    # TODO: verify against PostgreSQL users table
    if not email or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email and password required")

    # Stub: always succeed for demo
    token = create_access_token(data={"sub": email})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/register")
async def register(body: dict):
    """Register a new user."""
    email = body.get("email")
    password = body.get("password")
    # TODO: create user in PostgreSQL
    if not email or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email and password required")

    token = create_access_token(data={"sub": email})
    return {"access_token": token, "token_type": "bearer", "message": "User created"}
