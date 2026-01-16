import torch.nn as nn

from torch_audit.context import AuditContext, AuditState
from torch_audit.core import Phase, Severity

from torch_audit.validators.builtin.architecture import ArchitectureValidator


def test_redundant_bias_clean():
    """
    Case 1: Conv (bias=False) -> BN. Should PASS.
    Case 2: Conv (bias=True) -> ReLU. Should PASS.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Linear(10, 5, bias=True),
        nn.ReLU(),
    )

    state = AuditState(model=model, step=0, phase=Phase.STATIC)
    ctx = AuditContext(state)

    validator = ArchitectureValidator()
    findings = list(validator.check(ctx))

    assert len(findings) == 0


def test_redundant_bias_detected():
    """
    Case: Conv (bias=True) -> BN. Should FAIL (TA400).
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, bias=True),
        nn.BatchNorm2d(16)
    )

    state = AuditState(model=model, step=0, phase=Phase.STATIC)
    ctx = AuditContext(state)

    validator = ArchitectureValidator()
    findings = list(validator.check(ctx))

    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "TA400"
    assert f.severity == Severity.WARN
    assert "followed by Norm" in f.message
    # Entity should show the specific layer transition
    assert "->" in f.entity


def test_redundant_bias_linear():
    """
    Case: Linear (bias=True) -> BN. Should FAIL (TA400).
    """
    model = nn.Sequential(
        nn.Linear(32, 32, bias=True),
        nn.BatchNorm1d(32)
    )

    state = AuditState(model=model, step=0, phase=Phase.STATIC)
    ctx = AuditContext(state)

    validator = ArchitectureValidator()
    findings = list(validator.check(ctx))

    assert len(findings) == 1
    assert findings[0].rule_id == "TA400"