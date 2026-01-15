import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState, Phase
from torch_audit.validators.builtin.optimization import (
    TA402_WEIGHT_DECAY,
    TA403_EMBEDDING_DECAY,
    OptimizerValidator,
)
from torch_audit.validators.builtin.stability import TA104_NO_GRADS, StabilityValidator


def test_ta403_embedding_decay():
    model = nn.Embedding(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)

    validator = OptimizerValidator()
    # FIX: Use AuditState
    state = AuditState(model=model, optimizer=optimizer, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len(findings) == 1
    assert findings[0].rule_id == TA403_EMBEDDING_DECAY.id


def test_ta402_rms_norm_decay():
    if not hasattr(nn, "RMSNorm"):
        return

    model = nn.RMSNorm(10)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)

    validator = OptimizerValidator()
    # FIX: Use AuditState
    state = AuditState(model=model, optimizer=optimizer, phase=Phase.STATIC, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len(findings) >= 1
    assert findings[0].rule_id == TA402_WEIGHT_DECAY.id


def test_ta104_no_grads():
    model = nn.Linear(10, 10)

    validator = StabilityValidator()
    # FIX: Use AuditState
    state = AuditState(model=model, phase=Phase.BACKWARD, step=0)
    context = AuditContext(state)

    findings = list(validator.check(context))
    assert len(findings) == 1
    assert findings[0].rule_id == TA104_NO_GRADS.id