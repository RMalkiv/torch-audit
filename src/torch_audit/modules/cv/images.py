import torch
from typing import List, Dict, Any


def check_image_hygiene(batch: Any) -> List[Dict[str, Any]]:
    issues = []

    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        img = batch[0]
    else:
        img = batch

    if not isinstance(img, torch.Tensor):
        return issues

    # Heuristic: Images are usually 4D [B, C, H, W] or 3D [C, H, W]
    if img.dim() not in (3, 4):
        return issues

    min_val, max_val = img.min(), img.max()

    if max_val > 50.0:
        issues.append({
            "type": "CV Preprocessing",
            "layer": "Input Data",
            "message": f"Input values range from {min_val:.1f} to {max_val:.1f}. "
                       f"Neural networks expect inputs roughly in [0, 1] or [-1, 1]. "
                       f"Did you forget `ToTensor()` or division by 255?",
            "severity": "ERROR"
        })

    if img.std() < 1e-4:
        issues.append({
            "type": "CV Data Quality",
            "layer": "Input Data",
            "message": "Input batch has zero variance (blank/flat images). "
                       "Check your data loader or augmentation pipeline.",
            "severity": "WARNING"
        })

    return issues
