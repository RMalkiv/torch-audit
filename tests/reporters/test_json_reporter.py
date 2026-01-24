import io
import json

import pytest

from torch_audit.core import AuditResult, Finding, Phase, Rule, Severity
from torch_audit.reporters.json import JSONReporter


@pytest.fixture
def mock_result():
    # Setup: 2 Rules, 3 Findings (mixed order to test sorting)
    r1 = Rule("B001", "Rule B", "Desc", "Fix", "Cat", Severity.WARN)
    r2 = Rule("A001", "Rule A", "Desc", "Fix", "Cat", Severity.ERROR)

    # Finding for Rule B
    f1 = Finding(
        "B001", "Msg 1", Severity.WARN, module_path="mod.z", phase=Phase.FORWARD
    )
    # Finding for Rule A (Should come first in stable sort)
    f2 = Finding(
        "A001", "Msg 2", Severity.ERROR, module_path="mod.a", phase=Phase.STATIC
    )
    # Another Finding for Rule A
    f3 = Finding(
        "A001", "Msg 3", Severity.ERROR, module_path="mod.b", phase=Phase.STATIC
    )

    return AuditResult(
        findings=[f1, f2, f3],  # unsorted input
        exit_code=1,
        max_severity=Severity.ERROR,
        suppressed_count=2,
        rules={"B001": r1, "A001": r2},
    )


def test_json_reporter_stream(mock_result):
    """Verify writing to an open stream (stdout/buffer)."""
    stream = io.StringIO()
    reporter = JSONReporter(dest=stream)

    reporter.report(mock_result)

    # Reset and Parse
    stream.seek(0)
    data = json.load(stream)

    # Schema Checks
    assert data["meta"]["tool"] == "torch-audit"
    assert data["summary"]["exit_code"] == 1
    assert data["summary"]["suppressed_findings"] == 2

    # Verify Enum serialization
    assert data["summary"]["max_severity"] == "ERROR"


def test_json_reporter_file(mock_result, tmp_path):
    """Verify writing to a file path."""
    dest_file = tmp_path / "audit_report.json"
    reporter = JSONReporter(dest=str(dest_file))

    reporter.report(mock_result)

    assert dest_file.exists()

    with open(dest_file) as f:
        data = json.load(f)

    assert len(data["findings"]) == 3


def test_json_reporter_stability(mock_result):
    """
    Output findings should be deterministically sorted:
    By RuleID -> ModulePath -> Entity -> Message
    """
    stream = io.StringIO()
    JSONReporter(dest=stream).report(mock_result)
    stream.seek(0)
    data = json.load(stream)

    findings = data["findings"]

    # A001 should come before B001
    assert findings[0]["rule_id"] == "A001"
    assert findings[0]["module_path"] == "mod.a"

    assert findings[1]["rule_id"] == "A001"
    assert findings[1]["module_path"] == "mod.b"

    assert findings[2]["rule_id"] == "B001"


def test_json_reporter_rules_metadata(mock_result):
    """Only rules that appear in findings should be in the 'rules' block (or all known)."""
    stream = io.StringIO()
    JSONReporter(dest=stream).report(mock_result)
    stream.seek(0)
    data = json.load(stream)

    rules = data["rules"]
    assert "A001" in rules
    assert "B001" in rules

    # Check rule content serialization
    assert rules["A001"]["default_severity"] == "ERROR"
