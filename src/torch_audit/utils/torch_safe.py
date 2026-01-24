from typing import Any

import torch


def safe_detach(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


def is_meta_tensor(t: torch.Tensor) -> bool:
    return getattr(t, "is_meta", False)


def isfinite_all(t: torch.Tensor) -> bool:
    """
    Returns Python bool (note: device sync when tensor is on GPU).
    """
    if is_meta_tensor(t):
        return True
    if t.is_sparse:
        t = t._values()
    return torch.isfinite(t).all().item()


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    t = safe_detach(t)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
    }


def safe_norm(t: torch.Tensor) -> float | None:
    try:
        t = safe_detach(t)
        if t.is_sparse:
            t = t._values()
        return float(t.norm().item())
    except Exception:
        return None
