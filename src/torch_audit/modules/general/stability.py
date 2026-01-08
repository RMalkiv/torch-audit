import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from ...core.validator import Validator
from ...core.issue import AuditIssue


class StabilityValidator(Validator):
    """
    Dynamic checks for numerical stability.
    - Detects Mixed Precision (Autocast) usage.
    - Verifies Optimizer Epsilon is compatible with FP16.
    """

    def __init__(self, optimizer: Optional[torch.optim.Optimizer]):
        self.optimizer = optimizer
        self.hooks = []
        self.is_mixed_precision = False
        self._checked_once = False

    def _probe_hook(self, module, input, output):
        """
        Runs once to sniff the context.
        """
        if self._checked_once:
            return

        # Check an autocast context
        if torch.is_autocast_enabled():
            self.is_mixed_precision = True

        self._checked_once = True

    def attach(self, model: nn.Module):
        self._checked_once = False
        self.is_mixed_precision = False

        # Find just one leaf module to hook into
        for module in model.modules():
            has_children = any(True for _ in module.children())
            if not has_children:
                self.hooks.append(module.register_forward_hook(self._probe_hook))
                break

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        if self.optimizer is None:
            return issues

        # Check: Low Epsilon in Mixed Precision
        if self.is_mixed_precision:
            for i, group in enumerate(self.optimizer.param_groups):
                eps = group.get('eps', None)
                # (machine epsilon ~6e-5)
                if eps is not None and eps < 1e-7:
                    issues.append(AuditIssue(
                        type="Optimization Stability",
                        layer=f"Param Group {i}",
                        message=f"Optimizer epsilon ({eps}) is too low for Mixed Precision. "
                                f"Float16 has limited precision; small epsilons cause underflow/instability. "
                                f"Increase to 1e-7 or 1e-4.",
                        severity="WARNING"
                    ))
                    break  # Only report once

        return issues