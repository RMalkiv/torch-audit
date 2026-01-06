import torch
import torch.nn as nn
from functools import partial
from typing import Dict


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


class ActivationMonitor:
    def __init__(self, threshold: float = 0.90):
        self.hooks = []
        self.dead_counts: Dict[str, torch.Tensor] = {}
        self.threshold = threshold

    def _hook_fn(self, name: str, module: nn.Module, input, output):
        if hasattr(output, 'logits'):
            output = output.logits
        elif isinstance(output, dict):
            output = output.get('logits', list(output.values())[0])
        elif isinstance(output, tuple):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            return

        if output.numel() == 0:
            return

        sparsity = (output == 0).float().mean()

        self.dead_counts[name] = sparsity

    def attach(self, model: nn.Module, extra_classes=None):
        self.dead_counts = {}

        target_classes = DEFAULT_RECTIFIERS
        if extra_classes:
            target_classes = target_classes + tuple(extra_classes)

        for name, module in model.named_modules():
            if isinstance(module, target_classes):
                hook = module.register_forward_hook(partial(self._hook_fn, name))
                self.hooks.append(hook)

    def detach(self):
        """Safely removes all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_issues(self, model: nn.Module) -> list:
        """
        Analyzes the collected stats and generates reports.
        This runs ONCE at the end of the context manager, so CPU sync is fine here.
        """
        issues = []

        for name, sparsity_tensor in self.dead_counts.items():
            sparsity_val = sparsity_tensor.item()

            if sparsity_val > self.threshold:
                issues.append({
                    "type": "Activations",
                    "layer": name,
                    "message": f"{sparsity_val:.1%} zeros (Dead Neurons).",
                    "severity": "WARNING"
                })

        return issues
