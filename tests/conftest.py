import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Allow running `pytest` without needing an editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch_audit.context import AuditContext, AuditState  # noqa: E402
from torch_audit.core import Phase  # noqa: E402


@pytest.fixture
def simple_model() -> nn.Module:
    """A tiny model (CPU) for fast testing.

    Deterministic initialization for consistent gradient values.
    """
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    return model


@pytest.fixture
def audit_context(simple_model: nn.Module) -> AuditContext:
    """A standard AuditContext initialized with the simple_model."""
    state = AuditState(model=simple_model, step=0, phase=Phase.STATIC)
    return AuditContext(state)


@pytest.fixture
def training_context(simple_model: nn.Module) -> AuditContext:
    """An AuditContext where loss.backward() has been called.

    This ensures param.grad is not None.
    """
    for p in simple_model.parameters():
        p.requires_grad = True

    input_data = torch.randn(1, 2)
    target = torch.randn(1, 2)

    output = simple_model(input_data)
    loss = (output - target).sum()

    loss.backward()

    state = AuditState(
        model=simple_model, step=100, phase=Phase.BACKWARD, batch=input_data
    )
    return AuditContext(state)
