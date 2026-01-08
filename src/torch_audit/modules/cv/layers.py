import torch
import torch.nn as nn
from typing import List, Optional
from ...core.validator import Validator
from ...core.issue import AuditIssue


class ConvValidator(Validator):
    """
    Static checks for Convolutional layers.
    - Architecture: Detects even kernel sizes (alignment risks).
    - Optimization: Detects redundant biases before BatchNorm.
    - Capacity: Detects 'dead' filters (weights near zero).
    """

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        # 1. Scan for standard layer issues
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self._check_kernel_size(issues, name, module)
                self._check_dead_filters(issues, name, module)

        # 2. Scan for Redundant Bias (Conv+BN sequence)
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                self._check_bias_before_bn(issues, name, module)

        return issues

    def _check_kernel_size(self, issues: List[AuditIssue], name: str, module: nn.Conv2d):
        k = module.kernel_size
        if isinstance(k, int):
            k = (k, k)

        if (k[0] % 2 == 0 or k[1] % 2 == 0) and (k[0] > 1 or k[1] > 1):
            issues.append(AuditIssue(
                type="CV Architecture",
                layer=name,
                message=f"Using even kernel size {k}. "
                        f"Odd sizes (3x3, 5x5) are recommended for symmetric padding and alignment.",
                severity="INFO"
            ))

    def _check_dead_filters(self, issues: List[AuditIssue], name: str, module: nn.Conv2d):
        with torch.no_grad():
            filter_norms = module.weight.view(module.out_channels, -1).abs().sum(dim=1)
            dead_count = (filter_norms < 1e-6).sum().item()

        if dead_count > 0:
            percent_dead = dead_count / module.out_channels
            severity = "ERROR" if percent_dead > 0.5 else "WARNING"

            issues.append(AuditIssue(
                type="CV Capacity",
                layer=name,
                message=f"Found {dead_count} dead convolution filters ({percent_dead:.1%}). "
                        f"These filters have 0 weights. Check initialization or pruning.",
                severity=severity
            ))

    def _check_bias_before_bn(self, issues: List[AuditIssue], seq_name: str, seq: nn.Sequential):
        """
        Heuristic: If we see Conv2d(bias=True) followed immediately by BatchNorm, warn.
        """
        layers = list(seq.children())
        for i in range(len(layers) - 1):
            current = layers[i]
            nxt = layers[i + 1]

            if isinstance(current, nn.Conv2d) and isinstance(nxt, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if current.bias is not None:
                    issues.append(AuditIssue(
                        type="CV Optimization",
                        layer=f"{seq_name}[{i}]",
                        message=f"Conv2d has `bias=True` but is immediately followed by BatchNorm. "
                                f"The bias is mathematically redundant (cancelled by BN mean) "
                                f"and consumes unnecessary parameters/memory. Set `bias=False`.",
                        severity="WARNING"
                    ))