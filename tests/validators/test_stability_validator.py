import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState
from torch_audit.core import Phase, Severity
from torch_audit.validators.builtin.stability import StabilityValidator


def create_ctx(model, phase=Phase.BACKWARD):
    state = AuditState(model=model, step=0, phase=phase)
    return AuditContext(state)


def test_ta100_nan_inf_weights():
    """TA100: Weights NaN."""
    model = nn.Linear(10, 10)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    validator = StabilityValidator()
    findings = list(validator.check(create_ctx(model, Phase.STATIC)))

    assert len(findings) == 1
    assert findings[0].rule_id == "TA100"
    assert "contains NaNs" in findings[0].message


def test_ta102_grad_explosion():
    """TA102: Global Grad Norm Explosion."""
    model = nn.Linear(10, 10)

    for p in model.parameters():
        p.grad = torch.ones_like(p) * 1000.0

    validator = StabilityValidator(max_grad_norm=100.0)
    findings = list(validator.check(create_ctx(model)))

    ta102 = [f for f in findings if f.rule_id == "TA102"]
    assert len(ta102) == 1
    assert ta102[0].severity == Severity.WARN
    assert "exceeds threshold" in ta102[0].message


def test_ta103_dead_units():
    """TA103: Zero Gradients."""
    model = nn.Linear(10, 10)

    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)

    validator = StabilityValidator()
    findings = list(validator.check(create_ctx(model)))

    ta103 = [f for f in findings if f.rule_id == "TA103"]
    assert len(ta103) == 1
    assert "exactly zero" in ta103[0].message
    assert "grad:weight" in ta103[0].entity


def test_clean_model_pass():
    """Ensure clean model triggers nothing."""
    model = nn.Linear(10, 10)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1

    validator = StabilityValidator()
    findings = list(validator.check(create_ctx(model)))

    assert len(findings) == 0
