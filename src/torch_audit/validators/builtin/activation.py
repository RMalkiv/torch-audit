from collections.abc import Generator
from functools import partial

import torch
import torch.nn as nn

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

TA105_ACTIVATION_COLLAPSE = Rule(
    id="TA105",
    title="Activation Collapse (Dead Neurons)",
    description="A high percentage of neurons in a layer are outputting zero (Dead ReLU).",
    remediation="Check initialization, lower learning rate, or switch activation function (e.g. LeakyReLU).",
    category="Stability",
    default_severity=Severity.WARN,
)

RuleRegistry.register(TA105_ACTIVATION_COLLAPSE)


class ActivationValidator(BaseValidator):
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.hooks = []
        self.dead_counts: dict[str, float] = {}
        self.target_classes = (
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

    @property
    def rule(self):
        return TA105_ACTIVATION_COLLAPSE

    @property
    def supported_phases(self):
        # Runtime hooks collect activations during forward passes.
        # (A broader TRAIN phase isn't modeled in the current Phase enum.)
        return {Phase.FORWARD}

    def attach(self, model: nn.Module):
        """Registers forward hooks to monitor activations."""
        self.detach()  # Clear existing
        self.dead_counts = {}

        for name, module in model.named_modules():
            if isinstance(module, self.target_classes):
                hook = module.register_forward_hook(partial(self._hook_fn, name))
                self.hooks.append(hook)

    def detach(self):
        """Removes all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _hook_fn(self, name: str, module, input, output):
        # Handle tuples (e.g. RNNs) or dicts
        t_out = output
        if isinstance(output, tuple):
            t_out = output[0]
        elif isinstance(output, dict) and "logits" in output:
            t_out = output["logits"]

        if not isinstance(t_out, torch.Tensor):
            return

        if t_out.numel() > 0:
            # Simple sparsity check: count zeros
            is_zero = t_out == 0
            sparsity = is_zero.float().mean().item()
            # We store the latest sparsity; could be moving average in future
            self.dead_counts[name] = sparsity

    def on_phase_start(self, context: AuditContext) -> None:
        # Ensure we only ever report activations from the current forward.
        if context.phase == Phase.FORWARD:
            self.dead_counts.clear()

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        # If hooks haven't been attached, we can't do anything.
        # Ideally, AuditRunner calls attach() before the forward pass.

        for name, sparsity in self.dead_counts.items():
            if sparsity > self.threshold:
                yield Finding(
                    rule_id=TA105_ACTIVATION_COLLAPSE.id,
                    message=f"{sparsity:.1%} of neurons are dead (outputting 0).",
                    severity=Severity.WARN,
                    module_path=name,
                    step=context.step,
                    phase=context.phase,
                    metadata={"sparsity": sparsity},
                )

        # Clear stats after reporting to avoid stale alerts
        self.dead_counts.clear()
