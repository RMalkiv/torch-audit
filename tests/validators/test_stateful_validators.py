import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState, Phase
from torch_audit.validators.builtin.activation import (
    TA105_ACTIVATION_COLLAPSE,
    ActivationValidator,
)
from torch_audit.validators.builtin.graph import (
    TA500_UNUSED_LAYER,
    TA501_STATEFUL_REUSE,
    GraphValidator,
)


def test_ta105_activation_collapse():
    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.fill_(-1.0)

    validator = ActivationValidator(threshold=0.90)
    validator.attach(model)
    # Deterministic: with weight=-1 and bias=-1, an all-ones input guarantees
    # negative pre-activations, producing 100% zeros after ReLU.
    _ = model(torch.ones(2, 5))

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.FORWARD, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len(findings) == 1
    assert findings[0].rule_id == TA105_ACTIVATION_COLLAPSE.id

    validator.detach()


def test_ta500_zombie_layer():
    class ZombieModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(1, 1)
            self.unused = nn.Linear(1, 1)

        def forward(self, x):
            return self.used(x)

    model = ZombieModel()
    validator = GraphValidator()
    validator.attach(model)
    _ = model(torch.randn(1, 1))

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.FORWARD, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA500_UNUSED_LAYER.id]) == 1

    validator.detach()


def test_ta501_stateful_reuse():
    class ReuseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(5)

        def forward(self, x):
            return self.bn(self.bn(x))

    model = ReuseModel()
    model.train()

    validator = GraphValidator()
    validator.attach(model)
    _ = model(torch.randn(2, 5))

    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.FORWARD, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len([f for f in findings if f.rule_id == TA501_STATEFUL_REUSE.id]) == 1

    validator.detach()