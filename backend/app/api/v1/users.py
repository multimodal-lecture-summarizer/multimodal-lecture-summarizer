import uuid
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException, ForbiddenException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, create_pagination_metadata
from app.schemas.user import UserDTO
from app.api.deps import check_admin
from app.models.user import User
from app.core.constants import UserRole

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.get(
    "",
    response_model=BaseDTO[List[UserDTO]],
    summary="List all users (Admin only)",
    description="Retrieves a list of all registered users. Requires Admin permissions.",
)
def list_users(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    total = db.query(User).count()
    users = db.query(User).order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    return BaseDTO(
        success=True,
        data=[UserDTO.model_validate(u) for u in users],
        message="Users list retrieved successfully",
        metadata=create_pagination_metadata(
            limit=limit,
            offset=offset,
            total=total,
            count=len(users)
        ),
    )


@router.put(
    "/{user_id}/status",
    response_model=BaseDTO[UserDTO],
    summary="Toggle user active status (Admin only)",
    description="Enables or suspends a user account. Requires Admin permissions.",
)
def toggle_user_status(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    if user_id == current_user.user_id:
        raise ForbiddenException(message="You cannot block/unblock your own account")

    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise NotFoundException(message=f"User with ID {user_id} not found")

    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)

    return BaseDTO(
        success=True,
        data=UserDTO.model_validate(user),
        message=f"User status updated to {'active' if user.is_active else 'suspended'}",
    )


@router.put(
    "/{user_id}/role",
    response_model=BaseDTO[UserDTO],
    summary="Update user role (Admin only)",
    description="Changes the role of a user between Admin and User. Requires Admin permissions.",
)
def change_user_role(
    user_id: uuid.UUID,
    role: UserRole,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    if user_id == current_user.user_id:
        raise ForbiddenException(message="You cannot modify your own role")

    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise NotFoundException(message=f"User with ID {user_id} not found")

    user.role = role
    db.commit()
    db.refresh(user)

    return BaseDTO(
        success=True,
        data=UserDTO.model_validate(user),
        message=f"User role updated to {role}",
    )
