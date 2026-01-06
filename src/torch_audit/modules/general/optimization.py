import torch
import torch.nn as nn
from typing import List, Dict, Any


def check_weight_decay(model: nn.Module, optimizer: torch.optim.Optimizer) -> List[Dict[str, Any]]:
    """
    Checks if Weight Decay is correctly disabled for Biases and Norm layers.
    """
    if optimizer is None:
        return []

    issues = []

    param_to_name = {p: n for n, p in model.named_parameters()}

    norm_classes = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    norm_params = set()

    embedding_classes = (nn.Embedding, )
    embedding_params = set()

    for module in model.modules():
        if isinstance(module, norm_classes):
            for param in module.parameters():
                norm_params.add(param)
        elif isinstance(module, embedding_classes):
            for param in module.parameters():
                embedding_params.add(param)

    for i, group in enumerate(optimizer.param_groups):
        wd = group.get('weight_decay', 0.0)

        if wd == 0.0:
            continue

        for param in group['params']:
            if param not in param_to_name:
                continue

            name = param_to_name[param]

            if name.endswith(".bias"):
                issues.append({
                    "type": "Optimization (Weight Decay)",
                    "layer": name,
                    "message": f"Weight decay ({wd}) is enabled on a Bias term. Best practice is to set it to 0.0.",
                    "severity": "WARNING"
                })

            elif param in norm_params:
                issues.append({
                    "type": "Optimization (Weight Decay)",
                    "layer": name,
                    "message": f"Weight decay ({wd}) is enabled on a Normalization layer. This can reduce stability.",
                    "severity": "WARNING"
                })

            elif param in embedding_params:
                issues.append({
                    "type": "Optimization (Weight Decay)",
                    "layer": name,
                    "message": f"Weight decay ({wd}) is enabled on an Embedding layer. This can reduce stability.",
                    "severity": "WARNING"
                })

    return issues
