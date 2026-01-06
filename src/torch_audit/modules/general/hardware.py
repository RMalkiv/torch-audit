import torch.nn as nn


def check_tensor_core_alignment(model: nn.Module) -> list:
    """
    Scans Linear and Conv layers to ensure dimensions are multiples of 8.
    NVIDIA Tensor Cores require alignment to 8 (FP16) or 16 (INT8) for max speed.
    """
    issues = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features % 8 != 0 or module.out_features % 8 != 0:
                issues.append({
                    "type": "Hardware Efficiency",
                    "layer": name,
                    "message": f"Dimensions ({module.in_features} -> {module.out_features}) not divisible by 8. Tensor Cores will be inactive.",
                    "severity": "WARNING"
                })

        elif isinstance(module, nn.Conv2d):
            if module.in_channels % 8 != 0 or module.out_channels % 8 != 0:
                issues.append({
                    "type": "Hardware Efficiency",
                    "layer": name,
                    "message": f"Channels ({module.in_channels} -> {module.out_channels}) not divisible by 8. Tensor Cores will be inactive.",
                    "severity": "WARNING"
                })

    return issues
