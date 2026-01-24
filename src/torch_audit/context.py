from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from .core import Phase


@dataclass
class AuditState:
    model: torch.nn.Module
    step: int
    phase: Phase
    batch: Any | None = None
    optimizer: torch.optim.Optimizer | None = None


class AuditContext:
    def __init__(self, state: AuditState):
        self._state = state
        self._cache: dict[str, Any] = {}

    @property
    def model(self) -> torch.nn.Module:
        """
        Access to the raw PyTorch model.
        WARNING: Validators should treat this as read-only.
        Mutating the model during an audit can cause side effects in training.
        """
        return self._state.model

    @property
    def step(self) -> int:
        return self._state.step

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def batch(self) -> Any | None:
        """
        Optional input batch for runtime/data validators.
        Validators must treat this as read-only and must not store it.
        """
        return self._state.batch

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._state.optimizer

    def iter_param_info(self) -> Iterator[dict[str, Any]]:
        """
        Safe accessor for parameter metadata.
        Prevents accidental mutation of weights/grads and ensures JSON-safe types.
        """
        for name, param in self._state.model.named_parameters():
            yield {
                "name": name,
                "shape": tuple(param.shape),
                "dtype": str(param.dtype),
                "device": str(param.device),
                "requires_grad": param.requires_grad,
                "has_grad": param.grad is not None,
            }

    def iter_batch_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Yield (path, tensor) pairs found inside the batch.
        """
        b = self._state.batch
        if b is None:
            return
        yield from _walk_batch(b, "batch")


def _walk_batch(obj: Any, path: str) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _walk_batch(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_batch(v, f"{path}[{k!r}]")
