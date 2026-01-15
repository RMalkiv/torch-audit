from typing import List

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class ConvValidator(Validator):
    """
    Static checks for Convolutional layers (1D, 2D, 3D).
    - Architecture: Detects even kernel sizes (alignment risks).
    - Optimization: Detects redundant biases before BatchNorm.
    - Capacity: Detects 'dead' filters (weights near zero).
    """

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

        # 1. Scan for standard layer issues
        for name, module in model.named_modules():
            if isinstance(module, conv_types):
                self._check_kernel_size(issues, name, module)
                self._check_dead_filters(issues, name, module)

        # 2. Scan for Redundant Bias (Conv -> BN sequence)
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                self._check_bias_before_bn(issues, name, module)

        return issues

    def _check_kernel_size(
        self, issues: List[AuditIssue], name: str, module: nn.Module
    ):
        """
        Checks for even kernel sizes in any dimension.
        """
        k = module.kernel_size
        if isinstance(k, int):
            k = (k,)

        has_even_kernel = any((x % 2 == 0 and x > 1) for x in k)

        if has_even_kernel:
            issues.append(
                AuditIssue(
                    type="CV Architecture",
                    layer=name,
                    message=f"Using even kernel size {module.kernel_size}. "
                    f"Odd sizes (3, 5, 7) are recommended for symmetric padding and alignment.",
                    severity="INFO",
                )
            )

    def _check_dead_filters(
        self, issues: List[AuditIssue], name: str, module: nn.Module
    ):
        """
        Works for 1D [Out, In, L], 2D [Out, In, H, W], and 3D [Out, In, D, H, W].
        """
        with torch.no_grad():
            filter_norms = module.weight.view(module.out_channels, -1).abs().sum(dim=1)
            dead_count = (filter_norms < 1e-6).sum().item()

        if dead_count > 0:
            percent_dead = dead_count / module.out_channels
            severity = "ERROR" if percent_dead > 0.5 else "WARNING"

            issues.append(
                AuditIssue(
                    type="CV Capacity",
                    layer=name,
                    message=f"Found {dead_count} dead convolution filters ({percent_dead:.1%}). "
                    f"These filters have 0 weights. Check initialization or pruning.",
                    severity=severity,
                )
            )

    def _check_bias_before_bn(
        self, issues: List[AuditIssue], seq_name: str, seq: nn.Sequential
    ):
        """
        Detects Conv(bias=True) -> BN sequences for 1D, 2D, and 3D.
        """
        conv_classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        bn_classes = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LazyBatchNorm1d,
            nn.LazyBatchNorm2d,
            nn.LazyBatchNorm3d,
        )

        layers = list(seq.children())
        for i in range(len(layers) - 1):
            current = layers[i]
            nxt = layers[i + 1]

            if isinstance(current, conv_classes) and isinstance(nxt, bn_classes):
                if current.bias is not None:
                    issues.append(
                        AuditIssue(
                            type="CV Optimization",
                            layer=f"{seq_name}[{i}]",
                            message=f"{current.__class__.__name__} has `bias=True` but is immediately followed by {nxt.__class__.__name__}. "
                            f"The bias is mathematically redundant (cancelled by BN mean) "
                            f"and consumes unnecessary parameters/memory. Set `bias=False`.",
                            severity="WARNING",
                        )
                    )
