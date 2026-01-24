from collections import defaultdict
from collections.abc import Generator
from functools import partial

import torch.nn as nn

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

TA500_UNUSED_LAYER = Rule(
    id="TA500",
    title="Unused Layer (Zombie)",
    description="Layer is defined in the model but was never called during the forward pass.",
    remediation="Remove the layer or set `find_unused_parameters=True` (DDP).",
    category="Architecture",
    default_severity=Severity.ERROR,
)

TA501_STATEFUL_REUSE = Rule(
    id="TA501",
    title="Stateful Layer Reuse",
    description="A stateful layer (e.g. BatchNorm) was called multiple times.",
    remediation="Use distinct layers for each pass to avoid statistics corruption.",
    category="Architecture",
    default_severity=Severity.ERROR,
)

RuleRegistry.register(TA500_UNUSED_LAYER)
RuleRegistry.register(TA501_STATEFUL_REUSE)


class GraphValidator(BaseValidator):
    def __init__(self):
        self.hooks = []
        self.call_counts: dict[str, int] = defaultdict(int)
        self._seen_forward: bool = False
        # Atomic modules we don't recurse into
        self.atomic_modules = (nn.MultiheadAttention, nn.LSTM, nn.GRU, nn.RNN)

    @property
    def rule(self):
        return TA500_UNUSED_LAYER

    @property
    def rules(self):
        return [TA500_UNUSED_LAYER, TA501_STATEFUL_REUSE]

    @property
    def supported_phases(self):
        # Graph instrumentation is based on forward hooks.
        return {Phase.FORWARD}

    def attach(self, model: nn.Module):
        self.detach()
        self.call_counts.clear()
        self._seen_forward = False
        self._scan_and_hook(model, prefix="")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _scan_and_hook(self, module: nn.Module, prefix: str):
        # 1. Atomic Stop
        if isinstance(module, self.atomic_modules):
            self._register_hook(prefix, module)
            return

        # 2. Leaf Check
        has_children = any(True for _ in module.children())
        has_params = any(True for _ in module.parameters(recurse=False))

        if not has_children and has_params:
            self._register_hook(prefix, module)
            return

        # 3. Recurse
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            self._scan_and_hook(child, child_name)

    def _register_hook(self, name: str, module: nn.Module):
        self.call_counts[name] = 0
        # We bind 'name' to the hook
        hook = module.register_forward_hook(partial(self._hook_fn, name))
        self.hooks.append(hook)

    def _hook_fn(self, name: str, module, input, output):
        # Mark that we observed at least one forward call.
        self._seen_forward = True
        self.call_counts[name] += 1

    def on_phase_start(self, context: AuditContext) -> None:
        # Reset per-forward call counts so we don't accumulate counts across steps.
        if context.phase == Phase.FORWARD:
            self._seen_forward = False
            for k in self.call_counts:
                self.call_counts[k] = 0

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        # Only check this if we actually tracked something
        if not self.call_counts:
            return

        # Don't emit "unused" findings before we've observed a forward pass.
        # This avoids false positives when a user runs "data-only" audits.
        if not self._seen_forward:
            return

        name_to_module: dict[str, nn.Module] = dict(context.model.named_modules())

        for name, count in self.call_counts.items():
            module = name_to_module.get(name)
            if module is None:
                continue

            # TA500: Zombie
            if count == 0:
                yield Finding(
                    rule_id=TA500_UNUSED_LAYER.id,
                    message="Layer defined but NEVER called (Zombie).",
                    severity=Severity.ERROR,
                    module_path=name,
                    step=context.step,
                    phase=context.phase,
                )

            # TA501: Reuse
            elif count > 1:
                # Check if stateful (BN, Dropout, etc)
                is_stateful = hasattr(module, "track_running_stats") and getattr(
                    module, "track_running_stats", False
                )
                if is_stateful and module.training:
                    yield Finding(
                        rule_id=TA501_STATEFUL_REUSE.id,
                        message=f"Stateful layer called {count} times in one pass.",
                        severity=Severity.ERROR,
                        module_path=name,
                        step=context.step,
                        phase=context.phase,
                    )

        # Reset after reporting
        for k in self.call_counts:
            self.call_counts[k] = 0
