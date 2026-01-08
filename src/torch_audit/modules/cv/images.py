import torch
from typing import List, Any, Optional
from ...core.validator import Validator
from ...core.issue import AuditIssue


class ImageBatchValidator(Validator):
    """
    Checks Input Images for common CV data bugs.
    - Normalization: [0, 255] vs [0, 1].
    - Layout: Detects accidental NHWC (Channels Last) input.
    - Quality: Detects flat/empty images.
    """

    def check_data(self, batch: Any) -> List[AuditIssue]:
        issues = []

        # 1. Unwrap Batch (List/Tuple/Tensor/Dict)
        img = self._extract_image_tensor(batch)
        if img is None:
            return issues

        # 2. Check Dimensions
        # Heuristic: Images are 3D [C, H, W] or 4D [B, C, H, W]
        if img.dim() not in (3, 4):
            return issues

        # 3. Run Checks
        self._check_normalization(issues, img)
        self._check_flat_images(issues, img)
        self._check_channel_order(issues, img)

        return issues

    def _extract_image_tensor(self, batch: Any) -> Optional[torch.Tensor]:
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            return batch[0] if isinstance(batch[0], torch.Tensor) else None

        if isinstance(batch, dict):
            for k in ['pixel_values', 'images', 'image', 'inputs']:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    return batch[k]

            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.dim() in (3, 4):
                    return v

        return None

    def _check_normalization(self, issues: List[AuditIssue], img: torch.Tensor):
        if img.numel() == 0:
            return

        min_val, max_val = img.min().item(), img.max().item()

        if max_val > 50.0:
            issues.append(AuditIssue(
                type="CV Preprocessing",
                layer="Input Data",
                message=f"Input values range from {min_val:.1f} to {max_val:.1f}. "
                        f"Neural networks expect inputs roughly in [0, 1] or [-1, 1]. "
                        f"Did you forget `ToTensor()` or division by 255?",
                severity="ERROR"
            ))

    def _check_flat_images(self, issues: List[AuditIssue], img: torch.Tensor):
        if img.numel() < 2:
            return

        if img.std() < 1e-4:
            issues.append(AuditIssue(
                type="CV Data Quality",
                layer="Input Data",
                message="Input batch has near-zero variance (blank/flat images). "
                        "Check your data loader or augmentation pipeline.",
                severity="WARNING"
            ))

    def _check_channel_order(self, issues: List[AuditIssue], img: torch.Tensor):
        """
        Detects if input is likely NHWC (Batch, Height, Width, Channel).
        """
        if img.dim() != 4:
            return

        b, c, h, w = img.shape

        # Heuristic: Channels are usually small (1, 3, 4), Spatial dims >= 32
        if c >= 32 and w in (1, 3, 4):
            issues.append(AuditIssue(
                type="CV Data Layout",
                layer="Input Data",
                message=f"Input shape is {tuple(img.shape)}. "
                        f"PyTorch expects [Batch, Channel, Height, Width], but this looks like [Batch, Height, Width, Channel]. "
                        f"You may need `input.permute(0, 3, 1, 2)`.",
                severity="ERROR"
            ))
