import json
from typing import Any, List, Tuple


# PUBLIC_INTERFACE
def flatten_json(data: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """
    PUBLIC_INTERFACE
    Flatten an arbitrary JSON-serializable structure into dot-notation key/value pairs.

    - Uses dot notation for object properties (e.g., "user.address.city")
    - Uses dot notation with integer index for arrays (e.g., "users.0.name")
    - Produces only leaf entries (primitives, null, or empty object/array serialized compactly)
    - Path always reflects full hierarchy to preserve context

    Args:
        data (Any): Parsed JSON (dict, list, or primitive).
        prefix (str): The current path prefix (used internally during recursion).

    Returns:
        List[Tuple[str, str]]: List of (path, value_text) entries suitable for embedding and storage.
    """
    entries: List[Tuple[str, str]] = []

    def _is_primitive(x: Any) -> bool:
        return isinstance(x, (str, int, float, bool)) or x is None

    def _serialize_non_primitive(x: Any) -> str:
        # Compact JSON for non-primitive leaves (e.g., empty arrays/objects)
        try:
            return json.dumps(x, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return str(x)

    def _walk(node: Any, path: str):
        # Handle primitive
        if _is_primitive(node):
            # Normalize to string (preserve None as "null")
            if node is None:
                entries.append((path, "null"))
            elif isinstance(node, bool):
                entries.append((path, "true" if node else "false"))
            else:
                entries.append((path, str(node)))
            return

        # Handle dict
        if isinstance(node, dict):
            if not node:
                # Empty object counts as a leaf with serialized value
                entries.append((path, _serialize_non_primitive(node)))
                return
            for key, value in node.items():
                key_str = str(key)
                child_path = f"{path}.{key_str}" if path else key_str
                _walk(value, child_path)
            return

        # Handle list/tuple
        if isinstance(node, (list, tuple)):
            if not node:
                # Empty array counts as a leaf with serialized value
                entries.append((path, _serialize_non_primitive(node)))
                return
            for idx, value in enumerate(node):
                child_path = f"{path}.{idx}" if path else str(idx)
                _walk(value, child_path)
            return

        # Fallback: unknown type -> serialize
        entries.append((path, _serialize_non_primitive(node)))

    # Root can be object, array, or primitive
    root_path = prefix.strip(".")
    _walk(data, root_path if root_path else "")
    # Remove any entries with blank path by labeling as "$"
    normalized: List[Tuple[str, str]] = []
    for p, v in entries:
        normalized.append((p if p else "$", v))
    return normalized


# PUBLIC_INTERFACE
def format_entry_for_embedding(path: str, value: str) -> str:
    """
    PUBLIC_INTERFACE
    Build a compact, context-rich text line for embedding.

    Args:
        path (str): Dot-notation path (e.g., "orders.0.total")
        value (str): Leaf value (already serialized or stringified)

    Returns:
        str: Single string combining path and value (e.g., "orders.0.total = 123.45")
    """
    return f"{path} = {value}"
