import torch.nn as nn

from torch_audit.context import AuditContext, AuditState, Phase
from torch_audit.core import Severity
from torch_audit.validators.builtin.hardware import (
    TA200_TENSOR_CORE,
    TA201_CHANNELS_LAST,
    TA203_PRECISION,
    HardwareValidator,
)


def test_ta200_int8_alignment():
    model = nn.Linear(24, 24)
    validator = HardwareValidator()

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))

    int8_findings = [f for f in findings if f.rule_id == TA200_TENSOR_CORE.id and f.severity == Severity.INFO]
    assert len(int8_findings) == 1
    assert "INT8 misaligned" in int8_findings[0].message


def test_ta201_conv3d_channels_last():
    model = nn.Conv3d(16, 16, 3)
    validator = HardwareValidator()

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))

    conv3d_findings = [f for f in findings if f.rule_id == TA201_CHANNELS_LAST.id]
    assert len(conv3d_findings) >= 1


def test_ta203_precision_check():
    model = nn.Linear(10, 10).float()
    validator = HardwareValidator()

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))

    prec_findings = [f for f in findings if f.rule_id == TA203_PRECISION.id]
    assert len(prec_findings) == 1


def test_ta203_precision_check_bf16_clean():
    model = nn.Linear(10, 10).bfloat16()
    validator = HardwareValidator()

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA203_PRECISION.id]) == 0