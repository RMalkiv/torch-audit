import pytest
import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState
from torch_audit.core import Phase, Severity
from torch_audit.validators.builtin.data import DataValidator


def create_ctx(model, batch):
    state = AuditState(
        model=model,
        step=0,
        phase=Phase.FORWARD,
        batch=batch
    )
    return AuditContext(state)


def test_ta300_input_device_mismatch():
    """
    TA300: Model on GPU, Input on CPU -> ERROR.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    model = nn.Linear(10, 10).cuda()
    batch = torch.randn(5, 10).cpu()  # Mismatch

    validator = DataValidator()
    findings = list(validator.check(create_ctx(model, batch)))

    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "TA300"
    assert f.severity == Severity.ERROR
    assert "on cpu" in f.message.lower()


def test_ta301_data_range_scaling():
    """
    TA301: Values > 50.0 suggest missing [0,1] normalization.
    """
    model = nn.Identity()

    # Case 1: Unnormalized [0, 255]
    batch = torch.rand(4, 3, 32, 32) * 255.0

    validator = DataValidator()
    findings = list(validator.check(create_ctx(model, batch)))

    assert len(findings) > 0
    f = findings[0]
    assert f.rule_id == "TA301"
    assert f.severity == Severity.ERROR
    assert "range" in f.message

    # Case 2: Normalized [0, 1]
    batch_norm = torch.rand(4, 3, 32, 32)
    findings = list(validator.check(create_ctx(model, batch_norm)))
    assert len(findings) == 0


def test_ta302_flat_data():
    """
    TA302: Zero variance input (e.g. all zeros or all ones).
    """
    model = nn.Identity()

    # Blank image (all zeros)
    batch = torch.zeros(4, 3, 32, 32)

    validator = DataValidator()
    findings = list(validator.check(create_ctx(model, batch)))

    assert len(findings) > 0
    f = findings[0]
    assert f.rule_id == "TA302"
    assert f.severity == Severity.WARN
    assert "near-zero variance" in f.message
