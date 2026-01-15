from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Type

import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class GraphValidator(Validator):
    """
    Monitors the computation graph during execution.
    - Detects unused ("Zombie") layers (Critical for DDP).
    - Detects incorrect weight sharing (e.g. reusing BatchNorm).
    - Monitors graph complexity.
    """

    # Modules that internally use functional backends and bypass their children's hooks.
    # We must treat these as "Leaves" to avoid false positives on their submodules.
    DEFAULT_ATOMIC_MODULES = (
        nn.MultiheadAttention,
        nn.RNNBase,
        nn.LSTM,
        nn.GRU,
        nn.RNN,
        nn.LSTMCell,
        nn.GRUCell,
        nn.RNNCell,
    )

    def __init__(self, extra_atomic_modules: Optional[List[Type[nn.Module]]] = None):
        self.hooks = []
        self.call_counts: Dict[str, int] = defaultdict(int)

        self.atomic_modules = self.DEFAULT_ATOMIC_MODULES
        if extra_atomic_modules:
            self.atomic_modules += tuple(extra_atomic_modules)

    def attach(self, model: nn.Module):
        """
        Attaches hooks to 'leaf' old_modules, respecting Atomic boundaries.
        """
        self.call_counts.clear()
        self.hooks = []
        self._scan_and_hook(model, prefix="")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _scan_and_hook(self, module: nn.Module, prefix: str):
        """
        Recursively scans the model.
        Stops recursion if it hits an ATOMIC_MODULE (treating it as a leaf).
        """
        # 1. Check for Atomic Modules (MHA, RNNs)
        if isinstance(module, self.atomic_modules):
            self._register_hook(prefix, module)
            return

        # 2. Check for Standard Leaves
        has_children = any(True for _ in module.children())
        has_params = any(True for _ in module.parameters(recurse=False))

        if not has_children and has_params:
            self._register_hook(prefix, module)
            return

        # 3. Recurse into children
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            self._scan_and_hook(child, child_name)

    def _register_hook(self, name: str, module: nn.Module):
        """Helper to attach the hook and init the counter."""
        self.call_counts[name] = 0
        hook = module.register_forward_hook(partial(self._hook_fn, name))
        self.hooks.append(hook)

    def _hook_fn(self, name: str, module, input, output):
        self.call_counts[name] += 1

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        name_to_module = dict(model.named_modules())

        for name, count in self.call_counts.items():
            module = name_to_module.get(name)

            if module is None:
                continue

            # 1. Check for Zombie Layers
            if count == 0:
                issues.append(
                    AuditIssue(
                        type="DDP Safety",
                        layer=name,
                        message="Layer defined but NEVER called (Zombie). "
                        "In DDP, this causes deadlocks unless `find_unused_parameters=True`. "
                        "Delete this layer if unused.",
                        severity="ERROR",
                    )
                )

            # 2. Check for Reused Layers
            elif count > 1:
                has_running_stats = (
                    hasattr(module, "track_running_stats")
                    and getattr(module, "track_running_stats", False)
                    and getattr(module, "training", False)
                )

                if has_running_stats:
                    issues.append(
                        AuditIssue(
                            type="Logic Error",
                            layer=name,
                            message=f"Stateful layer called {count} times in one pass. "
                            f"This corrupts running statistics (mean/var). "
                            f"Use distinct copies for each pass.",
                            severity="ERROR",
                        )
                    )
                else:
                    issues.append(
                        AuditIssue(
                            type="Graph Complexity",
                            layer=name,
                            message=f"Layer called {count} times (Weight Tying detected). "
                            f"Ensure this is intentional (RNNs, Shared Embeddings).",
                            severity="INFO",
                        )
                    )

        return issues
