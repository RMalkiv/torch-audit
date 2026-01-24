from collections.abc import Generator

import torch
import torch.nn as nn

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

# --- Rules ---

TA400_REDUNDANT_BIAS = Rule(
    id="TA400",
    title="Redundant Bias before Norm",
    description="Linear/Conv layer has `bias=True` but is immediately followed by a Normalization layer.",
    remediation="Set `bias=False` on the layer to save memory and compute.",
    category="Architecture",
    default_severity=Severity.WARN,
)

TA404_EVEN_KERNEL = Rule(
    id="TA404",
    title="Even Kernel Size",
    description="Convolution uses an even kernel size (e.g., 2, 4). This can cause aliasing or shift artifacts.",
    remediation="Consider using odd kernel sizes (3, 5, 7) with symmetric padding.",
    category="Architecture",
    default_severity=Severity.INFO,
)

TA405_DEAD_FILTERS = Rule(
    id="TA405",
    title="Dead Convolution Filters",
    description="Some convolution filters have weights that are entirely zero (or near zero).",
    remediation="Check initialization or pruning logic. These filters contribute nothing.",
    category="Architecture",
    default_severity=Severity.WARN,
)

RuleRegistry.register(TA400_REDUNDANT_BIAS)
RuleRegistry.register(TA404_EVEN_KERNEL)
RuleRegistry.register(TA405_DEAD_FILTERS)


class ArchitectureValidator(BaseValidator):
    @property
    def rule(self):
        return TA400_REDUNDANT_BIAS

    @property
    def rules(self):
        return [TA400_REDUNDANT_BIAS, TA404_EVEN_KERNEL, TA405_DEAD_FILTERS]

    @property
    def supported_phases(self):
        # Architecture checks are static by nature.
        return {Phase.STATIC}

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        # 1. Sequential Checks (TA400)
        for name, module in context.model.named_modules():
            if isinstance(module, nn.Sequential):
                yield from self._check_sequential(name, module)

            # 2. Conv Checks (TA404, TA405)
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                yield from self._check_conv(name, module)

    def _check_sequential(
        self, seq_name: str, seq: nn.Sequential
    ) -> Generator[Finding, None, None]:
        prev_name = None
        prev_mod = None

        for name, mod in seq.named_children():
            if prev_mod is not None:
                if self._is_redundant(prev_mod, mod):
                    yield Finding(
                        rule_id=TA400_REDUNDANT_BIAS.id,
                        message=f"Layer '{prev_name}' has `bias=True` but is followed by Norm '{name}'.",
                        severity=TA400_REDUNDANT_BIAS.default_severity,
                        module_path=seq_name,
                        entity=f"{prev_name}->{name}",
                    )

            prev_name = name
            prev_mod = mod

    def _check_conv(
        self, name: str, layer: nn.Module
    ) -> Generator[Finding, None, None]:
        # --- TA404: Even Kernel ---
        # kernel_size can be int or tuple
        k = layer.kernel_size
        if isinstance(k, int):
            k = (k,)

        if any(d % 2 == 0 for d in k):
            yield Finding(
                rule_id=TA404_EVEN_KERNEL.id,
                message=f"Layer '{name}' has even kernel size {layer.kernel_size}.",
                severity=TA404_EVEN_KERNEL.default_severity,
                module_path=name,
                metadata={"kernel_size": str(layer.kernel_size)},
            )

        # --- TA405: Dead Filters ---
        # Weight shape: (Out, In/Groups, K...)
        # We want to check norms of each "Out" filter
        w = layer.weight
        if w is None:
            return

        with torch.no_grad():
            # Flatten all dims except the first (Out Channels)
            w_flat = w.reshape(w.shape[0], -1)
            # L2 norm per filter
            norms = torch.norm(w_flat, p=2, dim=1)

            # Threshold: extremely close to zero
            dead_mask = norms < 1e-6
            dead_count = dead_mask.sum().item()

            if dead_count > 0:
                yield Finding(
                    rule_id=TA405_DEAD_FILTERS.id,
                    message=f"Layer '{name}' has {dead_count} dead filters (near-zero weights).",
                    severity=TA405_DEAD_FILTERS.default_severity,
                    module_path=name,
                    metadata={
                        "dead_count": int(dead_count),
                        "total_filters": w.shape[0],
                    },
                )

    def _is_redundant(self, prev: nn.Module, curr: nn.Module) -> bool:
        has_bias = getattr(prev, "bias", None) is not None

        if not has_bias:
            return False

        is_linear_conv = isinstance(
            prev,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.Linear,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            ),
        )
        if not is_linear_conv:
            return False

        is_norm = isinstance(
            curr,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        )

        return is_norm
