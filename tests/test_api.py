from unittest.mock import patch

import pytest

from torch_audit.api import audit
from torch_audit.core import Finding, Rule, Severity
from torch_audit.validator import BaseValidator

# --- Helpers ---


class DummyValidator(BaseValidator):
    @property
    def rule(self):
        return Rule("DUMMY", "Title", "Desc", "Fix", "Cat", Severity.ERROR)

    def check(self, context):
        yield Finding(self.rule.id, "Found one", Severity.ERROR)


class FindingValidator(BaseValidator):
    def __init__(self, severity):
        self.sev = severity

    @property
    def rule(self):
        return Rule("TEST", "T", "D", "F", "C", Severity.ERROR)

    def check(self, context):
        yield Finding(self.rule.id, "msg", self.sev)


# --- Tests ---


def test_audit_defaults(simple_model):
    """
    Running audit() with minimal args should work and return a result.
    Passing validators=[] prevents loading defaults, ensuring speed.
    """
    result = audit(model=simple_model, validators=[])

    assert result.exit_code == 0
    assert len(result.findings) == 0
    assert result.max_severity == Severity.INFO


def test_audit_argument_normalization(simple_model):
    """
    String inputs for 'phase' and 'fail_level' should be converted to Enums.
    """
    audit(model=simple_model, phase="forward", fail_level="warn", validators=[])

    with pytest.raises(ValueError, match="Invalid phase"):
        audit(simple_model, phase="invalid_phase", strict=True)

    with pytest.raises(ValueError, match="Invalid fail_level"):
        audit(simple_model, fail_level="invalid_sev", strict=True)


def test_audit_execution_with_validators(simple_model):
    """
    Passing a custom validator list should execute them.
    """
    result = audit(
        model=simple_model, validators=[DummyValidator()], fail_level=Severity.ERROR
    )

    assert len(result.findings) == 1
    assert result.findings[0].rule_id == "DUMMY"
    assert result.exit_code == 1  # Found ERROR finding


def test_audit_baseline_passthrough(simple_model, tmp_path):
    """
    Verify baseline arguments are passed down to the internal config/runner.
    """
    baseline_file = tmp_path / "baseline.json"

    audit(
        model=simple_model,
        validators=[DummyValidator()],
        baseline_file=str(baseline_file),
        update_baseline=True,
    )

    assert baseline_file.exists()

    result = audit(
        model=simple_model,
        validators=[DummyValidator()],
        baseline_file=str(baseline_file),
        update_baseline=False,
        fail_level=Severity.ERROR,
    )

    assert len(result.findings) == 1
    assert result.exit_code == 0


def test_audit_strict_mode_invalid_inputs(simple_model):
    """strict=True must raise ValueError for invalid enums."""
    with pytest.raises(ValueError, match="Invalid phase"):
        audit(simple_model, phase="fake", strict=True, validators=[])

    with pytest.raises(ValueError, match="Invalid fail_level"):
        audit(simple_model, fail_level="CRITICAL", strict=True, validators=[])


def test_audit_fallback_is_error_level(simple_model):
    """
    Prove that strict=False falls back specifically to Severity.ERROR.
    """
    result_warn = audit(
        model=simple_model,
        fail_level="invalid_garbage",
        strict=False,
        validators=[FindingValidator(Severity.WARN)],
    )
    assert len(result_warn.findings) == 1
    assert result_warn.exit_code == 0

    result_err = audit(
        model=simple_model,
        fail_level="invalid_garbage",
        strict=False,
        validators=[FindingValidator(Severity.ERROR)],
    )
    assert len(result_err.findings) == 1
    assert result_err.exit_code == 1


def test_audit_show_report_lazy_import(simple_model):
    """
    Verify ConsoleReporter is NOT instantiated when show_report=False.
    This ensures we don't pay the import cost of Rich unnecessarily.
    """
    with patch("torch_audit.reporters.console.ConsoleReporter") as MockReporter:
        audit(model=simple_model, validators=[], show_report=False)
        MockReporter.assert_not_called()


def test_audit_reporter_logic(simple_model):
    """
    Verify show_report=True instantiates ConsoleReporter.
    """
    with patch("torch_audit.reporters.console.ConsoleReporter") as MockReporter:
        audit(model=simple_model, validators=[], show_report=True)
        MockReporter.assert_called_once()
