from collections.abc import Generator
from itertools import chain

import torch
import torch.nn as nn

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

# --- Rules ---

TA200_TENSOR_CORE = Rule(
    id="TA200",
    title="Tensor Core Alignment",
    description="Dimensions should be divisible by 8 (FP16) or 16 (INT8) for maximum throughput.",
    remediation="Pad dimensions to multiples of 8 (FP16) or 16 (INT8).",
    category="Performance",
    default_severity=Severity.WARN,
)

TA201_CHANNELS_LAST = Rule(
    id="TA201",
    title="Suboptimal Memory Layout",
    description="Conv2d/Conv3d layers are in NCHW/NCDHW format. Channels Last is faster on modern GPUs.",
    remediation="Use `.to(memory_format=torch.channels_last)` or `channels_last_3d`.",
    category="Performance",
    default_severity=Severity.WARN,
)

TA202_DEVICE_PLACEMENT = Rule(
    id="TA202",
    title="Model Device Placement",
    description="Model is split across devices (Split Brain) or on CPU when GPU is available.",
    remediation="Move all parameters to the correct device (e.g. `model.cuda()`).",
    category="Performance",
    default_severity=Severity.ERROR,
)

TA203_PRECISION = Rule(
    id="TA203",
    title="Precision Check (AMP)",
    description="Model seems to be using full FP32 precision. Modern GPUs run faster with AMP (FP16/BF16).",
    remediation="Use `torch.amp.autocast()` or convert weights to `bfloat16`.",
    category="Performance",
    default_severity=Severity.INFO,
)

RuleRegistry.register(TA200_TENSOR_CORE)
RuleRegistry.register(TA201_CHANNELS_LAST)
RuleRegistry.register(TA202_DEVICE_PLACEMENT)
RuleRegistry.register(TA203_PRECISION)


class HardwareValidator(BaseValidator):
    @property
    def rule(self):
        return TA202_DEVICE_PLACEMENT

    @property
    def rules(self):
        return [
            TA200_TENSOR_CORE,
            TA201_CHANNELS_LAST,
            TA202_DEVICE_PLACEMENT,
            TA203_PRECISION,
        ]

    @property
    def supported_phases(self):
        # Hardware / placement checks are effectively static and do not need to
        # run again during INIT by default (avoids duplicate findings).
        return {Phase.STATIC}

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        # 1. Device Placement (TA202)
        yield from self._check_devices(context.model)

        # 2. Tensor Core Alignment (TA200)
        yield from self._check_alignment(context.model)

        # 3. Memory Format (TA201)
        yield from self._check_memory_format(context.model)

        # 4. Precision Check (TA203)
        yield from self._check_precision(context.model)

    def _check_devices(self, model: nn.Module) -> Generator[Finding, None, None]:
        devices: set[str] = set()
        for tensor in chain(model.parameters(), model.buffers()):
            # Handle possible undefined devices or complex scenarios robustly
            if tensor.device.type == "cpu":
                devices.add("cpu")
            else:
                idx = tensor.device.index if tensor.device.index is not None else 0
                devices.add(f"{tensor.device.type}:{idx}")

        if not devices:
            return

        has_cpu = "cpu" in devices
        has_cuda = any("cuda" in d for d in devices)

        # Split Brain (Error)
        if has_cpu and has_cuda:
            yield Finding(
                rule_id=TA202_DEVICE_PLACEMENT.id,
                message=f"Model parameters are split across devices: {sorted(devices)}. "
                "This may cause RuntimeErrors or performance degradation.",
                severity=Severity.ERROR,
                metadata={"devices": list(devices)},
            )

        # Forgot Acceleration (Warning)
        elif len(devices) == 1 and has_cpu and torch.cuda.is_available():
            yield Finding(
                rule_id=TA202_DEVICE_PLACEMENT.id,
                message="Model is entirely on CPU, but a GPU is available.",
                severity=Severity.WARN,
            )

    def _check_alignment(self, model: nn.Module) -> Generator[Finding, None, None]:
        for name, mod in model.named_modules():
            dim_in, dim_out = None, None
            layer_type = ""

            if isinstance(mod, nn.Linear):
                dim_in, dim_out = mod.in_features, mod.out_features
                layer_type = "Linear"
            elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                dim_in, dim_out = mod.in_channels, mod.out_channels
                layer_type = mod.__class__.__name__

            if dim_in and dim_out:
                # Check 1: Divisible by 8 (FP16 Requirement) - WARNING
                if dim_in % 8 != 0 or dim_out % 8 != 0:
                    yield Finding(
                        rule_id=TA200_TENSOR_CORE.id,
                        message=f"{layer_type} dims ({dim_in} -> {dim_out}) not divisible by 8 (FP16 misaligned).",
                        severity=Severity.WARN,
                        module_path=name,
                        metadata={"in": dim_in, "out": dim_out, "misalignment": "fp16"},
                    )
                # Check 2: Divisible by 16 (INT8 Requirement) - INFO
                # We only flag this if it PASSED the % 8 check, to avoid double noise.
                elif dim_in % 16 != 0 or dim_out % 16 != 0:
                    yield Finding(
                        rule_id=TA200_TENSOR_CORE.id,
                        message=f"{layer_type} dims ({dim_in} -> {dim_out}) not divisible by 16 (INT8 misaligned).",
                        severity=Severity.INFO,
                        module_path=name,
                        metadata={"in": dim_in, "out": dim_out, "misalignment": "int8"},
                    )

    def _check_memory_format(self, model: nn.Module) -> Generator[Finding, None, None]:
        # Check Conv2d -> channels_last
        # Check Conv3d -> channels_last_3d

        found_suboptimal_2d = False
        found_suboptimal_3d = False

        for mod in model.modules():
            if isinstance(mod, nn.Conv2d) and not found_suboptimal_2d:
                if not mod.weight.is_contiguous(memory_format=torch.channels_last):
                    found_suboptimal_2d = True

            if isinstance(mod, nn.Conv3d) and not found_suboptimal_3d:
                if not mod.weight.is_contiguous(memory_format=torch.channels_last_3d):
                    found_suboptimal_3d = True

            if found_suboptimal_2d and found_suboptimal_3d:
                break

        if found_suboptimal_2d:
            yield Finding(
                rule_id=TA201_CHANNELS_LAST.id,
                message="Conv2d layers detected in NCHW format. Use `memory_format=torch.channels_last`.",
                severity=Severity.WARN,
                metadata={"type": "Conv2d"},
            )

        if found_suboptimal_3d:
            yield Finding(
                rule_id=TA201_CHANNELS_LAST.id,
                message="Conv3d layers detected in NCDHW format. Use `memory_format=torch.channels_last_3d`.",
                severity=Severity.WARN,
                metadata={"type": "Conv3d"},
            )

    def _check_precision(self, model: nn.Module) -> Generator[Finding, None, None]:
        total_params = 0
        fp32_params = 0

        for p in model.parameters():
            num = p.numel()
            total_params += num
            if p.dtype == torch.float32:
                fp32_params += num

        if total_params == 0:
            return

        fp32_ratio = fp32_params / total_params

        # If > 90% is FP32, suggest AMP
        if fp32_ratio > 0.90:
            yield Finding(
                rule_id=TA203_PRECISION.id,
                message=f"Model is {fp32_ratio:.1%} FP32. Consider using AMP or BFloat16 for speed.",
                severity=Severity.INFO,
                metadata={"fp32_ratio": fp32_ratio},
            )
