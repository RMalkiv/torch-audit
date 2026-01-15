import torch.nn as nn
import torch.optim as optim

from torch_audit.context import AuditContext, AuditState
from torch_audit.core import Phase
from torch_audit.validators.builtin.optimization import OptimizerValidator


# --- Helper to build context with optimizer ---
def create_opt_context(model, opt):
    state = AuditState(
        model=model,
        step=0,
        phase=Phase.STATIC,
        optimizer=opt
    )
    return AuditContext(state)


def test_optimizer_clean_adamw():
    """
    AdamW is generally safe (avoids TA401).
    To pass TA402 (Weight Decay on Bias), we use a model without bias
    or would need to separate param groups. Here we use bias=False for simplicity.
    """
    model = nn.Linear(10, 2, bias=False)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    ctx = create_opt_context(model, opt)
    validator = OptimizerValidator()
    findings = list(validator.check(ctx))

    assert len(findings) == 0


def test_ta401_adam_with_decay():
    """
    Using standard Adam with weight_decay > 0 should trigger TA401.
    """
    model = nn.Linear(10, 2)
    # Incorrect usage of Adam for decay
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ctx = create_opt_context(model, opt)
    validator = OptimizerValidator()
    findings = list(validator.check(ctx))

    # Expect TA401 (Adam vs AdamW)
    # TA402 might also trigger if we didn't separate params, but here we focus on TA401
    f_ids = [f.rule_id for f in findings]
    assert "TA401" in f_ids


def test_ta402_weight_decay_on_bias():
    """
    Weight decay should not be applied to bias parameters.
    """
    model = nn.Linear(10, 2, bias=True)  # Has 'weight' and 'bias'

    # Naively applying WD to all parameters
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    ctx = create_opt_context(model, opt)
    validator = OptimizerValidator()
    findings = list(validator.check(ctx))

    # Should find TA402 for the bias parameter
    ta402_findings = [f for f in findings if f.rule_id == "TA402"]
    assert len(ta402_findings) > 0

    msg = ta402_findings[0].message
    assert "enabled on Bias" in msg


def test_ta402_weight_decay_on_norm():
    """
    Weight decay should not be applied to BatchNorm/LayerNorm parameters.
    """
    model = nn.BatchNorm1d(10)

    # Naively applying WD to Norm
    opt = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)

    ctx = create_opt_context(model, opt)
    validator = OptimizerValidator()
    findings = list(validator.check(ctx))

    # Should find TA402 for weights and bias of BN
    ta402_findings = [f for f in findings if f.rule_id == "TA402"]
    assert len(ta402_findings) >= 1
    assert "enabled on Norm" in ta402_findings[0].message


def test_ta402_correct_parameter_groups():
    """
    If the user correctly splits param groups (decay vs no-decay),
    no findings should occur.
    """
    model = nn.Sequential(
        nn.Linear(10, 10, bias=True),
        nn.BatchNorm1d(10)
    )

    linear = model[0]
    bn = model[1]

    groups = [
        {"params": [linear.weight], "weight_decay": 0.01},
        {"params": [linear.bias, bn.weight, bn.bias], "weight_decay": 0.0},
    ]

    opt = optim.AdamW(groups, lr=1e-3)

    ctx = create_opt_context(model, opt)
    validator = OptimizerValidator()
    findings = list(validator.check(ctx))

    assert len(findings) == 0
