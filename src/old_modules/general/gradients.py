import math
from typing import List

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class GradientValidator(Validator):
    """
    Checks gradient norms AFTER loss.backward() has been run.
    - Detects Exploding Gradients.
    - Detects Vanishing/Zero Gradients.
    - Detects NaNs or Infs (Stability).
    """

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        # 1. Filter parameters that actually have gradients
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if not grads:
            issues.append(
                AuditIssue(
                    type="Gradient Dynamics",
                    layer="Global",
                    message="No gradients found. Did you forget `loss.backward()` or are all parameters frozen?",
                    severity="ERROR",
                )
            )
            return issues

        # 2. Compute Total Norm (Global Gradient Norm) efficiently
        root_device = grads[0].device
        total_norm_sq = torch.zeros(1, device=root_device)

        for g in grads:
            param_norm = g.norm(2)

            # Handle Model Parallelism: Ensure norm is on the accumulation device
            if param_norm.device != root_device:
                param_norm = param_norm.to(root_device)

            total_norm_sq += param_norm**2

        total_norm = total_norm_sq.sqrt().item()

        # 3. Check for NaNs or Infs (Critical Stability Issue)
        if math.isnan(total_norm) or math.isinf(total_norm):
            issues.append(
                AuditIssue(
                    type="Gradient Stability",
                    layer="Global",
                    message="Gradient Norm is NaN or Infinite. Weights are corrupted.",
                    severity="ERROR",
                )
            )
            return issues

        # 4. Check Exploding Gradients
        if total_norm > self.threshold:
            issues.append(
                AuditIssue(
                    type="Gradient Dynamics",
                    layer="Global",
                    message=f"Gradient Norm is Exploding ({total_norm:.2f} > {self.threshold}). "
                    f"Consider using `torch.nn.utils.clip_grad_norm_` or lowering learning rate.",
                    severity="WARNING",
                )
            )

        # 5. Check Zero Gradients
        elif total_norm == 0.0:
            issues.append(
                AuditIssue(
                    type="Gradient Dynamics",
                    layer="Global",
                    message="Gradient Norm is Zero. The model is not learning (Check frozen layers or detached graph).",
                    severity="ERROR",
                )
            )

        return issues
