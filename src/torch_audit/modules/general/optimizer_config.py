import torch
import torch.nn as nn
from typing import List, Optional

from ...core.validator import Validator
from ...core.issue import AuditIssue


class OptimizerConfigValidator(Validator):
    """
    Static checks for Optimizer configuration best practices.
    - Checks Weight Decay exclusion for Biases/Norms.
    - Suggests AdamW over Adam.
    """

    def __init__(self, optimizer: Optional[torch.optim.Optimizer]):
        self.optimizer = optimizer

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        if self.optimizer is None:
            return []

        issues = []
        self._check_adam_vs_adamw(issues)
        self._check_weight_decay_targets(issues, model)
        return issues

    def _check_adam_vs_adamw(self, issues: List[AuditIssue]):
        """
        Adam with weight_decay > 0 is usually inferior to AdamW.
        """
        if isinstance(self.optimizer, torch.optim.Adam):
            has_wd = any(group.get('weight_decay', 0.0) > 0 for group in self.optimizer.param_groups)

            if has_wd:
                issues.append(AuditIssue(
                    type="Optimizer Selection",
                    layer="Global",
                    message="Using `torch.optim.Adam` with `weight_decay > 0`. "
                            "Prefer `torch.optim.AdamW`. AdamW decouples weight decay from gradient updates, "
                            "usually yielding better generalization.",
                    severity="WARNING"
                ))

    def _check_weight_decay_targets(self, issues: List[AuditIssue], model: nn.Module):
        """
        Ensures Weight Decay is disabled (set to 0.0) for:
        - Biases (standard practice)
        - Normalization layers (LayerNorm, BatchNorm)
        - Embeddings
        """
        param_to_name = {p: n for n, p in model.named_parameters()}

        safe_params = set()

        # 1. Norms
        norm_classes = (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.RMSNorm,
            nn.GroupNorm, nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d
        )
        for module in model.modules():
            if isinstance(module, norm_classes):
                for param in module.parameters():
                    safe_params.add(param)

        # 2. Iterate Optimizer Groups
        for group in self.optimizer.param_groups:
            wd = group.get('weight_decay', 0.0)
            if wd == 0.0:
                continue

            for param in group['params']:
                if param not in param_to_name: continue

                name = param_to_name[param]
                issue_msg = None

                if name.endswith(".bias"):
                    issue_msg = f"Weight decay ({wd}) enabled on Bias. Set to 0.0."
                elif param in safe_params:
                    issue_msg = f"Weight decay ({wd}) enabled on Norm layer. Set to 0.0."
                else:
                    parent_mod = model
                    if '.' in name:
                        parent_path = name.rsplit('.', 1)[0]
                        try:
                            parent_mod = model.get_submodule(parent_path)
                        except AttributeError:
                            pass

                    if isinstance(parent_mod, nn.Embedding):
                        issue_msg = f"Weight decay ({wd}) enabled on Embedding. Consider sparse gradients."

                if issue_msg:
                    issues.append(AuditIssue(
                        type="Optimization (Weight Decay)",
                        layer=name,
                        message=issue_msg,
                        severity="WARNING"
                    ))
