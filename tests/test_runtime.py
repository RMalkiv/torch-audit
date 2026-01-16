import torch
import torch.nn as nn


def test_auditor_runs_stateful_validators_without_manual_attach():
    from torch_audit.runtime import Auditor
    from torch_audit.validators.builtin.activation import (
        TA105_ACTIVATION_COLLAPSE,
        ActivationValidator,
    )

    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.fill_(-1.0)

    auditor = Auditor(model, validators=[ActivationValidator(threshold=0.9)])

    with auditor:
        _ = auditor.forward(torch.ones(2, 5))
        result = auditor.finish()

    assert any(f.rule_id == TA105_ACTIVATION_COLLAPSE.id for f in result.findings)


def test_auditor_graph_validator_zombie_detection():
    from torch_audit.runtime import Auditor
    from torch_audit.validators.builtin.graph import TA500_UNUSED_LAYER, GraphValidator

    class ZombieModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(1, 1)
            self.unused = nn.Linear(1, 1)

        def forward(self, x):
            return self.used(x)

    model = ZombieModel()
    auditor = Auditor(model, validators=[GraphValidator()])
    with auditor:
        _ = auditor.forward(torch.randn(1, 1))
        result = auditor.finish()

    assert any(f.rule_id == TA500_UNUSED_LAYER.id for f in result.findings)
