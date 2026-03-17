"""
DataRef resolution: file paths, URIs, and in-memory object references.

Resolves DataRef schema objects into concrete Python values.  Supports
three kinds: ``path`` (local file loaded as JSON/NumPy), ``uri`` (remote
fetch), and ``object`` (keyed lookup in an in-process object store).
The ``register_object`` / ``resolve_data_ref`` pair allows passing large
arrays or pre-built objects into a JSON-configured model without
serialization.
"""

import json
import os
from typing import Any
from urllib.parse import urlparse

from darts.api.schemas import DataRef

_OBJECT_STORE: dict[str, Any] = {}


def register_object(key: str, obj: Any) -> None:
    """Register a Python object for later reference by DataRef."""
    _OBJECT_STORE[key] = obj


def get_object(key: str) -> Any:
    """Retrieve a previously registered object."""
    return _OBJECT_STORE[key]


def resolve_data_ref(
    ref: DataRef | dict[str, Any],
    *,
    base_path: str | None = None,
    object_store: dict[str, Any] | None = None,
) -> Any:
    """Resolve DataRef into in-memory data or objects."""
    if not isinstance(ref, DataRef):
        ref = DataRef.model_validate(ref)

    store = object_store if object_store is not None else _OBJECT_STORE

    if ref.kind == "object":
        if ref.value not in store:
            raise KeyError(f"Object reference '{ref.value}' not found")
        return store[ref.value]

    if ref.kind == "uri":
        parsed = urlparse(ref.value)
        if parsed.scheme in ("file", ""):
            path = parsed.path or parsed.netloc
        elif parsed.scheme in ("object", "obj"):
            key = parsed.netloc or parsed.path.lstrip("/")
            if key not in store:
                raise KeyError(f"Object reference '{key}' not found")
            return store[key]
        else:
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    else:
        path = ref.value

    if not os.path.isabs(path) and base_path:
        path = os.path.join(base_path, path)

    fmt = ref.format
    if fmt is None and path.lower().endswith(".json"):
        fmt = "json"

    if fmt == "binary":
        with open(path, "rb") as f:
            return f.read()
    if fmt == "text":
        with open(path, encoding=ref.encoding or "utf-8") as f:
            return f.read()

    # Default: JSON
    with open(path, encoding=ref.encoding or "utf-8") as f:
        return json.load(f)
