import pytest
import torch
import torch.nn as nn

from torch_audit.context import AuditContext, AuditState
from torch_audit.core import Phase, Severity
from torch_audit.validators.builtin.hardware import HardwareValidator


def create_ctx(model):
    state = AuditState(model=model, step=0, phase=Phase.STATIC)
    return AuditContext(state)


def test_ta200_tensor_core_alignment():
    """
    TA200: Dimensions should be multiples of 8.
    """
    # 1. Bad Model (Input 7 -> Output 7)
    bad_model = nn.Linear(7, 7)
    validator = HardwareValidator()
    findings = list(validator.check(create_ctx(bad_model)))

    # Fix: Filter specifically for TA200, ignoring potential TA202 warnings
    ta200_findings = [f for f in findings if f.rule_id == "TA200"]

    assert len(ta200_findings) > 0
    assert ta200_findings[0].severity == Severity.WARN
    assert "not divisible by 8" in ta200_findings[0].message

    # 2. Good Model (Input 16 -> Output 32)
    good_model = nn.Linear(16, 32)
    findings = list(validator.check(create_ctx(good_model)))

    # Ensure no TA200 is found
    ta200_findings = [f for f in findings if f.rule_id == "TA200"]
    assert len(ta200_findings) == 0


def test_ta201_channels_last():
    """
    TA201: Conv2d should ideally be in Channels Last format.
    """
    model = nn.Conv2d(16, 32, 3)

    validator = HardwareValidator()
    findings = list(validator.check(create_ctx(model)))

    ids = [f.rule_id for f in findings]
    assert "TA201" in ids

    model = model.to(memory_format=torch.channels_last)
    findings = list(validator.check(create_ctx(model)))

    ids = [f.rule_id for f in findings]
    assert "TA201" not in ids


def test_ta202_split_brain():
    """
    TA202: Model should not be split across CPU and GPU (if GPU available).
    """
    if not torch.cuda.is_available():
        pytest.skip("Skipping split-brain test (requires CUDA)")

    model = nn.Sequential(nn.Linear(10, 10).cpu(), nn.Linear(10, 10).cuda())

    validator = HardwareValidator()
    findings = list(validator.check(create_ctx(model)))

    ta202 = [f for f in findings if f.rule_id == "TA202"]
    assert len(ta202) == 1
    assert ta202[0].severity == Severity.ERROR
    assert "split across devices" in ta202[0].message


def test_ta202_cpu_warning():
    """
    TA202: Warn if model is CPU-only but CUDA is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("Skipping CPU warning test (requires CUDA)")

    model = nn.Linear(10, 10).cpu()

    validator = HardwareValidator()
    findings = list(validator.check(create_ctx(model)))

    ta202 = [f for f in findings if f.rule_id == "TA202"]
    assert len(ta202) == 1
    assert ta202[0].severity == Severity.WARN
    assert "entirely on CPU" in ta202[0].message
