import torch
import torch.nn as nn
from typing import List, Dict, Any


def check_gradients(model: nn.Module, threshold: float = 10.0) -> List[Dict[str, Any]]:
    """
    Checks gradient norms AFTER loss.backward() has been run.
    Optimized to minimize GPU-CPU synchronization points.
    """
    issues = []
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

    if not grads:
        return [{
            "type": "Gradient Dynamics",
            "layer": "Global",
            "message": "No gradients found. Did you forget `loss.backward()` or are all parameters frozen?",
            "severity": "ERROR"
        }]

    device = grads[0].device
    total_norm = torch.zeros(1, device=device)

    for g in grads:
        total_norm += g.norm(2) ** 2

    total_norm = total_norm.sqrt().item()

    if total_norm > threshold:
        issues.append({
            "type": "Gradient Dynamics",
            "layer": "Global",
            "message": f"Gradient Norm is Exploding ({total_norm:.2f}). Consider using 'torch.nn.utils.clip_grad_norm_'.",
            "severity": "WARNING"
        })
    elif total_norm == 0.0:
        issues.append({
            "type": "Gradient Dynamics",
            "layer": "Global",
            "message": "Gradient Norm is Zero. The model is not learning (Check frozen layers or detached graph).",
            "severity": "ERROR"
        })

    return issues
