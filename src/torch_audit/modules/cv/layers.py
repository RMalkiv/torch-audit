import torch.nn as nn
from typing import List, Dict, Any


def check_conv_layers(model: nn.Module) -> List[Dict[str, Any]]:
    issues = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filter_norms = module.weight.view(module.out_channels, -1).abs().sum(dim=1)

            dead_count = (filter_norms < 1e-6).sum().item()

            if dead_count > 0:
                percent_dead = dead_count / module.out_channels
                severity = "ERROR" if percent_dead > 0.5 else "WARNING"

                issues.append({
                    "type": "CV Capacity",
                    "layer": name,
                    "message": f"Found {dead_count} dead convolution filters ({percent_dead:.1%}). "
                               f"These filters have 0 weights and contribute nothing. "
                               f"Consider lowering learning rate or changing initialization.",
                    "severity": severity
                })

            k = module.kernel_size
            if isinstance(k, int): k = (k, k)

            if (k[0] % 2 == 0 or k[1] % 2 == 0) and (k[0] > 1 or k[1] > 1):
                issues.append({
                    "type": "CV Architecture",
                    "layer": name,
                    "message": f"Using even kernel size {k}. This can cause alignment/aliasing issues in padding. "
                               f"Odd sizes (3x3, 5x5) are recommended for symmetric padding.",
                    "severity": "INFO"
                })

    return issues
