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
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional context, such as pagination info or performance metrics",
    )
