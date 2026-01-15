import pytest

from torch_audit.config import AuditConfig, Suppression
from torch_audit.core import Finding, Severity

# --- 1. Test Failure Thresholds ---


def test_is_failure_logic():
    """
    AuditConfig.is_failure(s) should return True if s >= fail_level.
    """
    # Case A: Fail on ERROR (default)
    config = AuditConfig(fail_level=Severity.ERROR)
    assert not config.is_failure(Severity.INFO)
    assert not config.is_failure(Severity.WARN)
    assert config.is_failure(Severity.ERROR)
    assert config.is_failure(Severity.FATAL)

    # Case B: Fail on WARN
    config_warn = AuditConfig(fail_level=Severity.WARN)
    assert not config_warn.is_failure(Severity.INFO)
    assert config_warn.is_failure(Severity.WARN)
    assert config_warn.is_failure(Severity.ERROR)


# --- 2. Test Suppression Regex Compilation ---


def test_suppression_init_regex():
    """
    Suppression should compile valid regex and raise ValueError for invalid ones (fail-fast).
    """
    # Valid regex
    s_valid = Suppression(
        rule_id="TA001", reason="ignore", module_regex=r"^layers\.0\..*"
    )
    assert s_valid._regex is not None
    assert s_valid._regex.match("layers.0.weight")

    # Invalid regex -> Should crash loudly
    with pytest.raises(ValueError, match="Invalid suppression regex"):
        Suppression(
            rule_id="TA001", reason="ignore", module_regex="[unclosed-bracket"
        )


# --- 3. Test Suppression Matching Logic ---


def test_suppression_matching_rule_only():
    """
    If no regex is provided, it should suppress ALL findings for that rule_id.
    """
    suppression = Suppression(rule_id="TA001", reason="Noisy rule")

    # Matching ID
    f_match = Finding(rule_id="TA001", message="x", severity=Severity.ERROR)
    assert suppression.matches(f_match) is True

    # Non-matching ID
    f_diff = Finding(rule_id="TA002", message="x", severity=Severity.ERROR)
    assert suppression.matches(f_diff) is False


def test_suppression_matching_with_regex():
    """
    If regex IS provided, it must match both rule_id AND module_path.
    """
    suppression = Suppression(
        rule_id="TA001", reason="Ignore specific layer", module_regex=r"encoder\..*"
    )

    # Case 1: Match ID + Match Regex -> True
    f_hit = Finding(
        rule_id="TA001",
        message="msg",
        severity=Severity.WARN,
        module_path="encoder.layer1",
    )
    assert suppression.matches(f_hit) is True

    # Case 2: Match ID + Miss Regex -> False
    f_miss = Finding(
        rule_id="TA001",
        message="msg",
        severity=Severity.WARN,
        module_path="decoder.layer1",
    )
    assert suppression.matches(f_miss) is False

    # Case 3: Miss ID -> False (even if regex matches)
    f_wrong_id = Finding(
        rule_id="TA002",
        message="msg",
        severity=Severity.WARN,
        module_path="encoder.layer1",
    )
    assert suppression.matches(f_wrong_id) is False


# --- 4. Test Integration with AuditConfig ---


def test_audit_config_should_show():
    """
    AuditConfig.should_show() should return False if ANY suppression matches.
    """
    suppressions = [
        Suppression(rule_id="TA001", reason="ignore 1"),
        Suppression(rule_id="TA002", reason="ignore 2", module_regex="hidden"),
    ]
    config = AuditConfig(suppressions=suppressions)

    # TA001 is globally suppressed
    f1 = Finding(rule_id="TA001", message="x", severity=Severity.ERROR)
    assert config.should_show(f1) is False

    # TA002 is suppressed ONLY if module_path matches "hidden"
    f2_hidden = Finding(
        rule_id="TA002",
        message="x",
        severity=Severity.ERROR,
        module_path="my.hidden.layer",
    )
    f2_visible = Finding(
        rule_id="TA002",
        message="x",
        severity=Severity.ERROR,
        module_path="my.visible.layer",
    )

    assert config.should_show(f2_hidden) is False
    assert config.should_show(f2_visible) is True

    # TA003 is not suppressed
    f3 = Finding(rule_id="TA003", message="x", severity=Severity.ERROR)
    assert config.should_show(f3) is True