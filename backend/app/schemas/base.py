from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")


class CamelModel(BaseModel):
    """
    Base model configured to handle camelCase on the client side
    and snake_case on the backend side.
    Supports reading from SQLAlchemy objects (from_attributes=True).
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class BaseDTO(CamelModel, Generic[T]):
    """
    Standard Response DTO wrapper for all API responses.
    """

    success: bool = Field(
        default=True, description="Indicates whether the request was successful"
    )
    data: Optional[T] = Field(
        default=None, description="The data payload of the response"
    )
    error: Optional[Any] = Field(
        default=None,
        description="Detailed error details. Filled only when success is False",
    )
    code: int = Field(
        default=200, description="HTTP status code or application custom code"
    )
    message: str = Field(
        default="Success",
        description="A readable status message, helpful for debugging",
    )
    metadata: Optional[Any] = Field(
        default=None,
        description="Additional context, such as pagination info or performance metrics",
    )


class PaginationMetadata(CamelModel):
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")
    total_results: int = Field(..., description="Total number of matching results")
    limit: int = Field(..., description="Limit parameter for compatibility")
    offset: int = Field(..., description="Offset parameter for compatibility")
    count: int = Field(..., description="Number of items in current page")
    total: int = Field(..., description="Total number of items")
    
    # Optional status aggregation fields for admin screens
    completed: Optional[int] = Field(None, description="Total completed items")
    failed: Optional[int] = Field(None, description="Total failed items")
    processing: Optional[int] = Field(None, description="Total pending/processing items")


def create_pagination_metadata(
    limit: int,
    offset: int,
    total: int,
    count: int,
    completed: Optional[int] = None,
    failed: Optional[int] = None,
    processing: Optional[int] = None,
) -> PaginationMetadata:
    """Helper to calculate pages and build PaginationMetadata securely."""
    effective_limit = limit if limit > 0 else 10
    total_pages = (total + effective_limit - 1) // effective_limit
    current_page = (offset // effective_limit) + 1
    
    return PaginationMetadata(
        total_pages=total_pages,
        current_page=current_page,
        page_size=limit,
        total_results=total,
        limit=limit,
        offset=offset,
        count=count,
        total=total,
        completed=completed,
        failed=failed,
        processing=processing,
    )

