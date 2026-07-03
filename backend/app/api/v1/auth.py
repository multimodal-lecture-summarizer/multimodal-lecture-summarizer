from datetime import timedelta
from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import AlreadyExistsException, AuthException
from app.core.config import settings
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, UserCreate, UserLogin, UserDTO, Token
from app.api.deps import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_active_user,
)
from app.models.user import User

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.post(
    "/register",
    response_model=BaseDTO[UserDTO],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers a new user in the system with the provided email and password.",
)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Registers a new user account in PostgreSQL database."""
    # Check if email is already taken
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        raise AlreadyExistsException(
            message=f"The email address {user_in.email} is already registered."
        )

    # Hash password and create user
    hashed_password = get_password_hash(user_in.password)
    db_user = User(
        email=user_in.email,
        password_hash=hashed_password,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return BaseDTO(
        success=True,
        data=UserDTO.model_validate(db_user),
        code=201,
        message="User registered successfully",
    )


@router.post(
    "/login",
    response_model=BaseDTO[Token],
    summary="User login to retrieve JWT access token",
    description="Authenticates credentials and issues a JWT token. Handles JSON requests.",
)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    """Logs in user with email and password via JSON payload."""
    user = db.query(User).filter(User.email == login_in.email).first()
    if not user or not verify_password(login_in.password, user.password_hash):
        raise AuthException(message="Incorrect email or password")

    access_token = create_access_token(subject=user.user_id, role=user.role)
    return BaseDTO(
        success=True,
        data=Token(access_token=access_token, token_type="bearer"),
        message="Authentication successful",
    )


@router.post(
    "/login/oauth2",
    response_model=Token,
    summary="OAuth2 compatible token login (for Swagger UI)",
    description="Login using OAuth2 password flow form submission. Required by Swagger OpenAPI auth.",
)
def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate OAuth2 credentials. Form data based for Swagger integrations."""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise AuthException(message="Incorrect email or password")

    access_token = create_access_token(subject=user.user_id, role=user.role)
    return Token(access_token=access_token, token_type="bearer")


@router.get(
    "/me",
    response_model=BaseDTO[UserDTO],
    summary="Retrieve current user profile details",
    description="Returns profile details of the currently logged in user verified by JWT.",
)
def get_me(current_user: User = Depends(get_current_active_user)):
    """Retrieves current user details."""
    return BaseDTO(
        success=True,
        data=UserDTO.model_validate(current_user),
        message="Profile details retrieved successfully",
    )
