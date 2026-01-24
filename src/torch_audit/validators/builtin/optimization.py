from collections.abc import Generator

import torch
import torch.nn as nn

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

TA401_ADAMW = Rule(
    id="TA401",
    title="AdamW vs Adam",
    description="Using `Adam` with `weight_decay > 0` is often inferior to `AdamW`.",
    remediation="Switch to `torch.optim.AdamW` for decoupled weight decay.",
    category="Optimization",
    default_severity=Severity.WARN,
)

TA402_WEIGHT_DECAY = Rule(
    id="TA402",
    title="Weight Decay on Norm/Bias",
    description="Weight decay is applied to Normalization layers or Bias terms.",
    remediation="Set `weight_decay=0.0` for these parameter groups.",
    category="Optimization",
    default_severity=Severity.WARN,
)

TA403_EMBEDDING_DECAY = Rule(
    id="TA403",
    title="Weight Decay on Embeddings",
    description="Weight decay is applied to Embedding layers. This breaks sparse gradients and is often harmful.",
    remediation="Set `weight_decay=0.0` for embeddings or use sparse optimizers.",
    category="Optimization",
    default_severity=Severity.WARN,
)

RuleRegistry.register(TA401_ADAMW)
RuleRegistry.register(TA402_WEIGHT_DECAY)
RuleRegistry.register(TA403_EMBEDDING_DECAY)


class OptimizerValidator(BaseValidator):
    @property
    def rule(self):
        return TA401_ADAMW

    @property
    def rules(self):
        return [TA401_ADAMW, TA402_WEIGHT_DECAY, TA403_EMBEDDING_DECAY]

    @property
    def supported_phases(self):
        # Optimizer configuration checks should run once when an optimizer is
        # available (INIT). Running in STATIC commonly causes duplicates.
        return {Phase.INIT}

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        opt = context.optimizer
        if opt is None:
            return

        # --- TA401: Adam vs AdamW ---
        if type(opt) is torch.optim.Adam:
            has_wd = any(
                group.get("weight_decay", 0.0) > 0 for group in opt.param_groups
            )
            if has_wd:
                yield Finding(
                    rule_id=TA401_ADAMW.id,
                    message="Found `torch.optim.Adam` with `weight_decay > 0`. "
                    "This applies L2 regularization, not decoupled weight decay.",
                    severity=TA401_ADAMW.default_severity,
                )

        # --- TA402 & TA403: Weight Decay Targets ---

        # 1. Identify Safe Params (Norms) & Embedding Params
        norm_classes = [
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ]
        # Support RMSNorm if available (PyTorch 2.4+)
        if hasattr(nn, "RMSNorm"):
            norm_classes.append(nn.RMSNorm)

        safe_params = set()
        embedding_params = set()

        param_to_name = {p: n for n, p in context.model.named_parameters()}

        for module in context.model.modules():
            if isinstance(module, tuple(norm_classes)):
                for param in module.parameters():
                    safe_params.add(param)
            elif isinstance(module, nn.Embedding):
                for param in module.parameters():
                    embedding_params.add(param)

        # 2. Check Groups
        for i, group in enumerate(opt.param_groups):
            wd = group.get("weight_decay", 0.0)
            if wd == 0.0:
                continue

            for param in group["params"]:
                if param not in param_to_name:
                    continue

                name = param_to_name[param]

                # Check Bias
                if name == "bias" or name.endswith(".bias"):
                    yield Finding(
                        rule_id=TA402_WEIGHT_DECAY.id,
                        message=f"Weight decay ({wd}) enabled on Bias parameter '{name}'.",
                        severity=TA402_WEIGHT_DECAY.default_severity,
                        module_path="optimizer",
                        entity=f"group_{i}::{name}",
                    )

                # Check Norms
                elif param in safe_params:
                    yield Finding(
                        rule_id=TA402_WEIGHT_DECAY.id,
                        message=f"Weight decay ({wd}) enabled on Norm parameter '{name}'.",
                        severity=TA402_WEIGHT_DECAY.default_severity,
                        module_path="optimizer",
                        entity=f"group_{i}::{name}",
                    )

                # Check Embeddings (TA403)
                elif param in embedding_params:
                    yield Finding(
                        rule_id=TA403_EMBEDDING_DECAY.id,
                        message=f"Weight decay ({wd}) enabled on Embedding parameter '{name}'.",
                        severity=TA403_EMBEDDING_DECAY.default_severity,
                        module_path="optimizer",
                        entity=f"group_{i}::{name}",
                    )
