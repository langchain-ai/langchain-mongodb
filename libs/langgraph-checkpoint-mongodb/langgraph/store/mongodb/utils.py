from typing import Any


def extract(d: dict[str, Any], keys: list[str]) -> list[Any]:
    """Extract values from nested dicts where keys may be of form parent.child"""

    def get_nested_value(data: Any, path_parts: list[str]) -> Any:
        for part in path_parts:
            if isinstance(data, dict):
                data = data.get(part)
            else:
                return None
        return data

    result = []
    for key in keys:
        path_parts = key.split(".")
        value = get_nested_value(d, path_parts)
        if value is not None:
            result.append(value)
    return result


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)
