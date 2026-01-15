from itertools import chain
from typing import List, Set

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class TensorCoreValidator(Validator):
    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        """
        Scans Linear and Conv layers to ensure dimensions align with Tensor Core requirements.
        - BF16/FP16 usually requires alignment to 8.
        - INT8 usually requires alignment to 16.
        """
        issues = []

        for name, module in model.named_modules():
            # Check Linear Layers
            if isinstance(module, nn.Linear):
                self._check_layer(
                    issues,
                    name,
                    "Linear",
                    dim_in=module.in_features,
                    dim_out=module.out_features,
                    label_in="in_features",
                    label_out="out_features",
                )

            # Check Convolutional Layers
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._check_layer(
                    issues,
                    name,
                    module.__class__.__name__,
                    dim_in=module.in_channels,
                    dim_out=module.out_channels,
                    label_in="in_channels",
                    label_out="out_channels",
                )

        return issues

    def _check_layer(
        self,
        issues: List[AuditIssue],
        layer_name: str,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        label_in: str,
        label_out: str,
    ):
        """
        Helper to check dimensions for both 8-bit (FP16) and 16-bit (INT8) alignment.
        """
        # 1. FP16 Alignment (Multiple of 8)
        if dim_in % 8 != 0 or dim_out % 8 != 0:
            issues.append(
                AuditIssue(
                    type="Tensor Core Alignment",
                    layer=layer_name,
                    message=f"{layer_type} dims ({dim_in} -> {dim_out}) not divisible by 8. "
                    f"BF16/FP16 Tensor Cores may be inactive. "
                    f"(Note: Classification heads are common offenders, consider padding).",
                    severity="WARNING",
                )
            )
            # return early to avoid double noise.
            return

        # 2. INT8 Alignment (Multiple of 16)
        if dim_in % 16 != 0 or dim_out % 16 != 0:
            issues.append(
                AuditIssue(
                    type="INT8 Optimization",
                    layer=layer_name,
                    message=f"{layer_type} dims ({dim_in} -> {dim_out}) divisible by 8 but not 16. "
                    f"For best INT8 quantization performance (e.g. on A100/H100), align to 16.",
                    severity="INFO",
                )
            )


class MemoryLayoutValidator(Validator):
    """
    Checks if Conv layers are using the optimal memory format:
    - Conv2d -> 'Channels Last' (NHWC)
    - Conv3d -> 'Channels Last 3D' (NDHWC)
    This typically yields 20-30% speedup on modern NVIDIA GPUs (Volta+).
    """

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        has_conv2d = False
        has_conv3d = False
        optimal_2d = False
        optimal_3d = False

        for _, module in model.named_modules():
            # Check Conv2d
            if isinstance(module, nn.Conv2d):
                has_conv2d = True
                if module.weight.is_contiguous(memory_format=torch.channels_last):
                    optimal_2d = True

            # Check Conv3d
            elif isinstance(module, nn.Conv3d):
                has_conv3d = True
                if module.weight.is_contiguous(memory_format=torch.channels_last_3d):
                    optimal_3d = True

            if (has_conv2d and optimal_2d) and (not has_conv3d or optimal_3d):
                break

        if has_conv2d and not optimal_2d:
            issues.append(
                AuditIssue(
                    type="Hardware Efficiency",
                    layer="Global",
                    message="Conv2d layers detected in NCHW format. Convert to Channels Last (NHWC) for "
                    "speedup: `model = model.to(memory_format=torch.channels_last)`",
                    severity="WARNING",
                )
            )

        if has_conv3d and not optimal_3d:
            issues.append(
                AuditIssue(
                    type="Hardware Efficiency",
                    layer="Global",
                    message="Conv3d layers detected in NCDHW format. Convert to Channels Last 3D (NDHWC) for "
                    "speedup: `model = model.to(memory_format=torch.channels_last_3d)`",
                    severity="WARNING",
                )
            )

        return issues


class PrecisionValidator(Validator):
    """
    Checks if the model is stored in FP32. Suggests Mixed Precision or BF16.
    """

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []
        param_count = 0
        fp32_count = 0

        for p in model.parameters():
            param_count += 1
            if p.dtype == torch.float32:
                fp32_count += 1

        if param_count > 0 and (fp32_count / param_count) > 0.9:
            issues.append(
                AuditIssue(
                    type="Hardware Efficiency",
                    layer="Global",
                    message="Model weights are >90% FP32. On modern GPUs, use AMP (Automatic Mixed Precision) "
                    "or BFloat16 (`torch.set_float32_matmul_precision('high')`) "
                    "for 2x-3x throughput.",
                    severity="INFO",
                )
            )

        return issues


class DevicePlacementValidator(Validator):
    """
    Checks for two common device placement issues:
    1. 'Split Brain': Model has tensors on both CPU and GPU (causes RuntimeErrors).
    2. 'Forgot Acceleration': Model is entirely on CPU but a GPU is available.
    """

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []
        devices: Set[str] = set()

        # Scan every parameter and buffer to find all used devices
        for tensor in chain(model.parameters(), model.buffers()):
            devices.add(
                f"{tensor.device.type}:{tensor.device.index}"
                if tensor.device.index is not None
                else tensor.device.type
            )

        if not devices:
            return issues

        # Logic 1: Check for "Split Brain" (CPU mixed with GPU)
        has_cpu = any("cpu" in d for d in devices)
        has_cuda = any("cuda" in d for d in devices)

        if has_cpu and has_cuda:
            issues.append(
                AuditIssue(
                    type="Device Mismatch",
                    layer="Global",
                    message=f"Model is split across devices: {devices}. "
                    f"PyTorch cannot mix CPU and GPU tensors in the same operation (RuntimeError). "
                    f"Check for layers initialized after .to(device) or buffers not registered correctly.",
                    severity="ERROR",
                )
            )
            return issues

        # Logic 2: Check for "All CPU" when GPU is available
        if len(devices) == 1 and has_cpu and torch.cuda.is_available():
            issues.append(
                AuditIssue(
                    type="Hardware Acceleration",
                    layer="Global",
                    message="Model is entirely on CPU, but a GPU is available. "
                    "Did you forget `model.cuda()` or `model.to(device)`?",
                    severity="WARNING",
                )
            )

        return issues
