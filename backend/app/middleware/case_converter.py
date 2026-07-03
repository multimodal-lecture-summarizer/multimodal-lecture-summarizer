import re
from typing import Callable, Any, Dict
from fastapi.routing import APIRoute
from fastapi import Request, Response


def camel_to_snake(name: str) -> str:
    """Converts camelCase to snake_case."""
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def snake_to_camel(name: str) -> str:
    """Converts snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class CamelCaseAPIRoute(APIRoute):
    """
    Custom APIRoute class that intercepts requests and responses to convert
    query parameters from camelCase to snake_case.
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            # Reconstruct query parameters from camelCase to snake_case
            if request.query_params:
                query_params_dict = dict(request.query_params)
                snake_query_params = {}
                for key, val in query_params_dict.items():
                    snake_key = camel_to_snake(key)
                    snake_query_params[snake_key] = val

                # Modify the request query_params in-place
                # Note: request.scope["query_string"] holds the raw bytes of the query string.
                # Rebuilding it allows FastAPI's dependency injection to bind correctly.
                from urllib.parse import urlencode

                new_query_string = urlencode(snake_query_params).encode("utf-8")
                request.scope["query_string"] = new_query_string

            response: Response = await original_route_handler(request)
            return response

        return custom_route_handler


def convert_keys_to_camel(data: Any) -> Any:
    """
    Recursively converts dict keys from snake_case to camelCase.
    Useful for formatting raw dictionary responses before returning them to client.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = snake_to_camel(key) if isinstance(key, str) else key
            new_dict[new_key] = convert_keys_to_camel(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_camel(item) for item in data]
    return data
