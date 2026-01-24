from typing import Any

import torch


def json_safe(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively converts objects to JSON-serializable primitives.
    """
    if depth >= max_depth:
        return str(obj)

    # Primitives
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    # Tensors
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return {
            "type": "Tensor",
            "shape": tuple(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }

    # Torch Types
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)

    # Collections
    if isinstance(obj, (list, tuple)):
        return [json_safe(x, depth + 1, max_depth) for x in obj]

    if isinstance(obj, dict):
        return {str(k): json_safe(v, depth + 1, max_depth) for k, v in obj.items()}

    # Objects (best effort)
    try:
        if hasattr(obj, "__dict__"):
            d = vars(obj)
            if d:
                return json_safe(d, depth + 1, max_depth)
    except Exception:
        pass

    return str(obj)


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Converts tensors to metadata dicts and ensures JSON safety.
    """
    return json_safe(meta)
