from collections.abc import Generator

import torch

from ...context import AuditContext
from ...core import Finding, Phase, Rule, Severity
from ...registry import RuleRegistry
from ...validator import BaseValidator

# --- Rules ---

TA300_INPUT_DEVICE = Rule(
    id="TA300",
    title="Input Device Mismatch",
    description="Input batch is on a different device (e.g., CPU) than the model (GPU).",
    remediation="Move data to the correct device using `.to(device, non_blocking=True)`.",
    category="Data Integrity",
    default_severity=Severity.ERROR,
)

TA301_DATA_RANGE = Rule(
    id="TA301",
    title="Suspicious Data Range",
    description="Input data range is suspicious (e.g., [0, 255] for float inputs).",
    remediation="Normalize inputs to [0, 1] or [-1, 1].",
    category="Data Integrity",
    default_severity=Severity.ERROR,
)

TA302_DATA_FLAT = Rule(
    id="TA302",
    title="Flat/Empty Data",
    description="Input batch has near-zero variance (blank images/tokens).",
    remediation="Check data loader, augmentation pipeline, or file integrity.",
    category="Data Integrity",
    default_severity=Severity.WARN,
)

TA303_SUSPICIOUS_LAYOUT = Rule(
    id="TA303",
    title="Suspicious Input Layout (NHWC vs NCHW)",
    description="Input tensor shape looks like NHWC (Channels Last), but PyTorch expects NCHW.",
    remediation="Ensure input is permuted correctly: `x.permute(0, 3, 1, 2)`.",
    category="Data Integrity",
    default_severity=Severity.WARN,
)

TA304_TINY_BATCH = Rule(
    id="TA304",
    title="Tiny Batch Size",
    description="Batch size is very small (e.g. < 8) while using BatchNorm. This causes training instability.",
    remediation="Increase batch size or switch to GroupNorm/LayerNorm.",
    category="Data Integrity",
    default_severity=Severity.WARN,
)

TA305_INVALID_INPUT = Rule(
    id="TA305",
    title="Invalid Input Values",
    description="Input contains invalid values (e.g., negative integers for embeddings).",
    remediation="Check data preprocessing and vocabulary mapping.",
    category="Data Integrity",
    default_severity=Severity.ERROR,
)

RuleRegistry.register(TA300_INPUT_DEVICE)
RuleRegistry.register(TA301_DATA_RANGE)
RuleRegistry.register(TA302_DATA_FLAT)
RuleRegistry.register(TA303_SUSPICIOUS_LAYOUT)
RuleRegistry.register(TA304_TINY_BATCH)
RuleRegistry.register(TA305_INVALID_INPUT)


class DataValidator(BaseValidator):
    def __init__(self) -> None:
        # Cache whether the current model contains BatchNorm layers.
        # This avoids repeatedly scanning model.modules() for every tensor in a batch.
        self._has_bn_cached_model_id: int | None = None
        self._has_bn_cached: bool = False

    def _model_has_batchnorm(self, model: torch.nn.Module) -> bool:
        mid = id(model)
        if mid != self._has_bn_cached_model_id:
            self._has_bn_cached_model_id = mid
            self._has_bn_cached = any(
                isinstance(
                    m,
                    (
                        torch.nn.BatchNorm1d,
                        torch.nn.BatchNorm2d,
                        torch.nn.BatchNorm3d,
                    ),
                )
                for m in model.modules()
            )
        return self._has_bn_cached

    @property
    def rule(self):
        return TA300_INPUT_DEVICE

    @property
    def rules(self):
        return [
            TA300_INPUT_DEVICE,
            TA301_DATA_RANGE,
            TA302_DATA_FLAT,
            TA303_SUSPICIOUS_LAYOUT,
            TA304_TINY_BATCH,
            TA305_INVALID_INPUT,
        ]

    @property
    def supported_phases(self):
        # Data checks are meaningful during the forward phase (real batches).
        return {Phase.FORWARD}

    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        if context.batch is None:
            return

        # 1. Check Device Mismatch (TA300)
        try:
            model_device = next(context.model.parameters()).device
        except StopIteration:
            model_device = None

        # We iterate safely looking for tensors
        for path, tensor in context.iter_batch_tensors():
            if model_device is not None and tensor.device != model_device:
                if tensor.numel() > 1 and tensor.dtype in (
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ):
                    yield Finding(
                        rule_id=TA300_INPUT_DEVICE.id,
                        message=f"Input tensor '{path}' is on {tensor.device}, but model is on {model_device}.",
                        severity=Severity.ERROR,
                        entity=path,
                        metadata={
                            "tensor_device": str(tensor.device),
                            "model_device": str(model_device),
                        },
                    )

            # 2. Check Data Quality (TA301, TA302, TA303)
            if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
                yield from self._check_quality(tensor, path)
                yield from self._check_layout(tensor, path)

            yield from self._check_hygiene(tensor, path, context.model)

    def _check_quality(
        self, t: torch.Tensor, path: str
    ) -> Generator[Finding, None, None]:
        if t.numel() == 0:
            return

        if t.numel() > 0:
            max_val = t.max().item()
            min_val = t.min().item()

            if max_val > 50.0:
                yield Finding(
                    rule_id=TA301_DATA_RANGE.id,
                    message=f"Input values range from {min_val:.1f} to {max_val:.1f}. Expected approx [0, 1].",
                    severity=Severity.ERROR,
                    entity=path,
                )

        # TA302: Flat Check
        if t.numel() > 1:
            std = t.std().item()
            if std < 1e-5:
                yield Finding(
                    rule_id=TA302_DATA_FLAT.id,
                    message="Input tensor has near-zero variance (blank/flat data).",
                    severity=Severity.WARN,
                    entity=path,
                    metadata={"std": std},
                )

    def _check_layout(
        self, t: torch.Tensor, path: str
    ) -> Generator[Finding, None, None]:
        # Heuristic for TA303:
        # If 4D tensor (N, C, H, W)
        # But dim 1 (C) is large (>=32) and dim 3 (W) is small (<=4)
        # It is likely (N, H, W, C) -- i.e. NHWC loaded as NCHW
        if t.ndim == 4:
            N, C, H, W = t.shape
            if C >= 32 and W <= 4:
                # Strong likelihood of NHWC confusion (e.g. C=3 interpreted as W=3)
                yield Finding(
                    rule_id=TA303_SUSPICIOUS_LAYOUT.id,
                    message=f"Input shape {tuple(t.shape)} looks suspicious (NHWC loaded as NCHW?). "
                    f"Channel dim ({C}) is large, Width dim ({W}) is small.",
                    severity=Severity.WARN,
                    entity=path,
                    metadata={"shape": tuple(t.shape)},
                )

    def _check_hygiene(
        self, t: torch.Tensor, path: str, model: torch.nn.Module
    ) -> Generator[Finding, None, None]:
        # TA304: Tiny Batch (only if BatchNorm is present)
        # We perform a lightweight check for BatchNorm existence
        if t.ndim > 0 and t.shape[0] < 8:
            if self._model_has_batchnorm(model):
                yield Finding(
                    rule_id=TA304_TINY_BATCH.id,
                    message=f"Tiny batch size ({t.shape[0]}) detected with BatchNorm layers.",
                    severity=Severity.WARN,
                    entity=path,
                    metadata={"batch_size": t.shape[0]},
                )

        # TA305: Negative Integers
        if t.dtype in (torch.long, torch.int32, torch.int16):
            if (t < 0).any():
                yield Finding(
                    rule_id=TA305_INVALID_INPUT.id,
                    message="Found negative integer inputs. Embedding indices must be positive.",
                    severity=Severity.ERROR,
                    entity=path,
                )
