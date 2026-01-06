import torch
from typing import Any, List, Dict

def check_input_hygiene(batch: Any, config: Dict) -> List[Dict[str, Any]]:
    issues = []
    float_thresh = config.get("float_threshold", 10.0)
    batch_thresh = 8 if config.get("check_batch_size", True) else 0

    def _inspect_tensor(t: torch.Tensor, name: str):
        if not isinstance(t, torch.Tensor): return

        if t.dim() > 0 and t.shape[0] < batch_thresh:
            issues.append({
                "type": "Data Hygiene",
                "layer": name,
                "message": f"Batch size is tiny ({t.shape[0]}). BatchNorm is unstable on micro-batches. "
                           f"(Note: Gradient Accumulation does NOT fix BatchNorm stats).",
                "severity": "WARNING"
            })

        if torch.isnan(t).any() or torch.isinf(t).any():
            issues.append({
                "type": "Data Stability",
                "layer": name,
                "message": "Input contains NaNs or Infs. Training will crash.",
                "severity": "ERROR"
            })
            return

        if t.dtype in [torch.long, torch.int32, torch.int16]:
            if (t < 0).any():
                issues.append({
                    "type": "Data Validity",
                    "layer": name,
                    "message": "Found negative integer inputs. Embedding indices must be positive.",
                    "severity": "ERROR"
                })
            return

        if t.dtype in [torch.float32, torch.float16, torch.float64]:
            max_val = t.abs().max().item()

            if max_val > float_thresh:
                issues.append({
                    "type": "Data Scaling",
                    "layer": name,
                    "message": f"Input values are large (max abs: {max_val:.1f}). "
                               f"Neural nets expect inputs ~ N(0,1). Values > {float_thresh} can cause instability.",
                    "severity": "WARNING"
                })

    if isinstance(batch, torch.Tensor):
        _inspect_tensor(batch, "Batch (Single Tensor)")

    elif isinstance(batch, (tuple, list)):
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                _inspect_tensor(item, f"Batch Item [{i}]")

    elif isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                _inspect_tensor(v, f"Batch Key '{k}'")

    return issues
