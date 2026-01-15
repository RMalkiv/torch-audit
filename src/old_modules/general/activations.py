from functools import partial
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator

DEFAULT_RECTIFIERS = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.ELU,
    nn.Threshold,
    nn.Hardswish,
    nn.Hardsigmoid,
    nn.SiLU,
)


class ActivationValidator(Validator):
    """
    Monitors activation sparsity during the forward pass.
    Detects "Dead Neurons" (layers where >90% of outputs are zero).
    """

    def __init__(
        self,
        threshold: float = 0.90,
        extra_classes: Optional[List[Type[nn.Module]]] = None,
    ):
        self.threshold = threshold
        self.hooks = []
        self.dead_counts: Dict[str, float] = {}

        self.target_classes = DEFAULT_RECTIFIERS
        if extra_classes:
            self.target_classes = self.target_classes + tuple(extra_classes)

    def attach(self, model: nn.Module):
        """
        Registers forward hooks on all activation layers.
        """
        self.dead_counts = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, self.target_classes):
                # We use partial to bind the layer name to the hook
                hook = module.register_forward_hook(partial(self._hook_fn, name))
                self.hooks.append(hook)

    def detach(self):
        """Safely removes all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        """
        Analyzes the sparsity stats collected during the forward pass.
        """
        issues = []

        for name, sparsity_val in self.dead_counts.items():
            if sparsity_val > self.threshold:
                issues.append(
                    AuditIssue(
                        type="Activation Collapse",
                        layer=name,
                        message=f"{sparsity_val:.1%} of neurons are dead (outputting 0). "
                        f"High sparsity reduces model capacity. "
                        f"Check initialization or learning rate.",
                        severity="WARNING",
                    )
                )

        return issues

    def _hook_fn(self, name: str, module: nn.Module, input, output):
        """
        Calculates the percentage of zeros in the output tensor.
        """
        # 1. Unwrap the output (handle Transformers/Tuples)
        t_out = self._unwrap_output(output)

        if t_out is None:
            return

        if t_out.numel() == 0:
            return

        # 2. Calculate Sparsity (Ratio of Zeros)
        is_zero = t_out == 0
        sparsity = is_zero.float().mean().item()

        self.dead_counts[name] = sparsity

    def _unwrap_output(self, output) -> Optional[torch.Tensor]:
        """
        Helper to extract the main tensor from various output formats.
        """
        if isinstance(output, torch.Tensor):
            return output

        # Handle HuggingFace ModelOutput or custom objects with .logits
        if hasattr(output, "logits"):
            return output.logits

        # Handle Dictionaries
        if isinstance(output, dict):
            return output.get(
                "logits",
                next((v for v in output.values() if isinstance(v, torch.Tensor)), None),
            )

        # Handle Tuples
        if isinstance(output, (tuple, list)):
            # Usually the first element is the main output
            return (
                output[0]
                if len(output) > 0 and isinstance(output[0], torch.Tensor)
                else None
            )

        return None
