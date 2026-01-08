import torch.nn as nn
from functools import partial
from collections import defaultdict
from typing import Dict, List, Any


class GraphMonitor:
    def __init__(self):
        self.hooks = []
        self.call_counts: Dict[str, int] = defaultdict(int)

    def _hook_fn(self, name: str, module, input, output):
        self.call_counts[name] += 1

    def attach(self, model: nn.Module):
        """
        Attaches hooks to all 'leaf' modules that have parameters.
        Resets counters to ensure a fresh audit.
        """
        self.call_counts.clear()

        for name, module in model.named_modules():
            has_children = any(True for _ in module.children())
            has_params = any(True for _ in module.parameters(recurse=False))

            if not has_children and has_params:
                self.call_counts[name] = 0

                hook = module.register_forward_hook(partial(self._hook_fn, name))
                self.hooks.append(hook)

    def detach(self):
        """Removes hooks but KEEPS the stats for reporting."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_issues(self, model: nn.Module) -> List[Dict[str, Any]]:
        issues = []

        name_to_module = dict(model.named_modules())

        for name, count in self.call_counts.items():
            module = name_to_module.get(name)

            if count == 0:
                issues.append({
                    "type": "DDP Safety",
                    "layer": name,
                    "message": "Layer defined but NEVER called (Zombie). "
                               "In DDP, this causes deadlocks unless `find_unused_parameters=True` (which is slow). "
                               "Delete this layer if unused.",
                    "severity": "ERROR"
                })

            elif count > 1:
                has_running_stats = (
                    hasattr(module, 'track_running_stats') and
                    module.track_running_stats and
                    module.training
                )
                if has_running_stats:
                    issues.append({
                        "type": "Logic Error",
                        "layer": name,
                        "message": (
                            f"Stateful Normalization layer called {count} times in one pass. "
                            "This corrupts running statistics (mean/var). "
                            "Use distinct Normalization layers or distinct copies."
                        ),
                        "severity": "ERROR"
                    })

                else:
                    issues.append({
                        "type": "Graph Complexity",
                        "layer": name,
                        "message": f"Layer called {count} times. "
                                   f"Ensure this is intentional (e.g. RNN or Shared Embeddings). "
                                   f"In DDP, this requires specific bucket configuration.",
                        "severity": "INFO"
                    })
        return issues
