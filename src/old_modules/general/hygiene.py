from typing import Any, List

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class InputHygieneValidator(Validator):
    """
    Checks input data batches for common issues:
    - NaNs/Infs (Stability)
    - Tiny batch sizes (BatchNorm instability)
    - Negative inputs for embeddings
    - Large values (Scaling issues)
    """

    def __init__(self, float_threshold: float = 10.0, check_batch_size: bool = True):
        self.float_threshold = float_threshold
        self.check_batch_size = check_batch_size
        self.batch_thresh = 8 if check_batch_size else 0

        # We default to True to be safe (warn if we don't know).
        # This flag is updated when `auditor.audit_static()` is called.
        self.uses_batchnorm = True

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        """
        Scans the model to determine if BatchNorm is used.
        This allows us to suppress 'Tiny Batch Size' warnings for models
        that use LayerNorm/GroupNorm (where small batches are fine).
        """
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        self.uses_batchnorm = any(isinstance(m, bn_types) for m in model.modules())
        return []

    def check_data(self, batch: Any) -> List[AuditIssue]:
        issues = []

        if isinstance(batch, torch.Tensor):
            self._inspect_tensor(issues, batch, "Batch (Single Tensor)")

        elif isinstance(batch, (tuple, list)):
            for i, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    self._inspect_tensor(issues, item, f"Batch Item [{i}]")

        elif isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    self._inspect_tensor(issues, v, f"Batch Key '{k}'")

        return issues

    def _inspect_tensor(self, issues: List[AuditIssue], t: torch.Tensor, name: str):
        """
        Runs hygiene checks on a single tensor found within the batch.
        """
        if not isinstance(t, torch.Tensor):
            return

        # 1. Check Batch Size
        # Only relevant if the model actually uses BatchNorm.
        if self.uses_batchnorm and t.dim() > 0 and t.shape[0] < self.batch_thresh:
            issues.append(
                AuditIssue(
                    type="Data Hygiene",
                    layer=name,
                    message=f"Batch size is tiny ({t.shape[0]}). BatchNorm is unstable on micro-batches. "
                    f"(Note: Gradient Accumulation does NOT fix BatchNorm stats).",
                    severity="WARNING",
                )
            )

        # 2. Check for NaNs or Infs
        if torch.isnan(t).any() or torch.isinf(t).any():
            issues.append(
                AuditIssue(
                    type="Data Stability",
                    layer=name,
                    message="Input contains NaNs or Infs. Training will crash.",
                    severity="ERROR",
                )
            )
            return

        # 3. Check Integer Inputs
        if t.dtype in [torch.long, torch.int32, torch.int16]:
            if (t < 0).any():
                issues.append(
                    AuditIssue(
                        type="Data Validity",
                        layer=name,
                        message="Found negative integer inputs. Embedding indices must be positive.",
                        severity="ERROR",
                    )
                )
            return

        # 4. Check Float Scaling
        if t.dtype in [torch.float32, torch.float16, torch.float64]:
            max_val = t.abs().max().item()

            if max_val > self.float_threshold:
                issues.append(
                    AuditIssue(
                        type="Data Scaling",
                        layer=name,
                        message=f"Input values are large (max abs: {max_val:.1f}). "
                        f"Values > {self.float_threshold} can cause instability.",
                        severity="WARNING",
                    )
                )
