import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState, Phase
from torch_audit.validators.builtin.architecture import (
    TA404_EVEN_KERNEL,
    TA405_DEAD_FILTERS,
    ArchitectureValidator,
)
from torch_audit.validators.builtin.data import (
    TA303_SUSPICIOUS_LAYOUT,
    DataValidator,
)


def test_ta303_suspicious_layout():
    t = torch.randn(1, 64, 64, 3)

    validator = DataValidator()
    # FIX: Use AuditState
    state = AuditState(
        model=nn.Linear(1, 1), batch={"img": t}, phase=Phase.FORWARD, step=0
    )
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA303_SUSPICIOUS_LAYOUT.id]) == 1


def test_ta404_even_kernel():
    model = nn.Conv2d(16, 16, kernel_size=4)

    validator = ArchitectureValidator()
    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA404_EVEN_KERNEL.id]) == 1


def test_ta405_dead_filters():
    model = nn.Conv2d(3, 4, kernel_size=3)
    with torch.no_grad():
        model.weight[0].zero_()

    validator = ArchitectureValidator()
    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA405_DEAD_FILTERS.id]) == 1
