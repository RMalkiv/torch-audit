import io
import json

import pytest

from torch_audit.core import AuditResult, Finding, Rule, Severity
from torch_audit.reporters.sarif import SARIFReporter


@pytest.fixture
def mock_result():
    r1 = Rule("TA001", "Error Rule", "Desc", "Fix", "Cat", Severity.ERROR)
    r2 = Rule("TA002", "Info Rule", "Desc", "Fix", "Cat", Severity.INFO)

    f1 = Finding(
        "TA001", "Bad thing", Severity.ERROR, module_path="layer1", entity="weight"
    )
    f2 = Finding("TA002", "Info thing", Severity.INFO, module_path="layer2")

    return AuditResult(
        findings=[f1, f2],
        exit_code=1,
        max_severity=Severity.ERROR,
        rules={"TA001": r1, "TA002": r2},
    )


def test_sarif_reporter_stream(mock_result):
    """Verify valid SARIF JSON output to stream."""
    stream = io.StringIO()
    reporter = SARIFReporter(dest=stream)

    reporter.report(mock_result)

    stream.seek(0)
    sarif = json.load(stream)

    # Basic SARIF 2.1.0 headers
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "torch-audit"


def test_sarif_reporter_file(mock_result, tmp_path):
    """Verify valid SARIF output to file."""
    dest_file = tmp_path / "audit.sarif"
    reporter = SARIFReporter(dest=str(dest_file))

    reporter.report(mock_result)

    assert dest_file.exists()
    with open(dest_file) as f:
        sarif = json.load(f)

    assert len(sarif["runs"][0]["results"]) == 2


def test_sarif_fingerprints(mock_result):
    """
    GitHub Code Scanning requires 'partialFingerprints' for stable tracking.
    We check for both 'stableFingerprint' and 'partialFingerprints'.
    """
    stream = io.StringIO()
    SARIFReporter(dest=stream).report(mock_result)
    stream.seek(0)
    sarif = json.load(stream)

    result_obj = sarif["runs"][0]["results"][0]

    fps = result_obj["fingerprints"]
    partial = result_obj["partialFingerprints"]

    assert "stableFingerprint" in fps
    assert "torchAuditFingerprint" in partial
    # The fingerprint logic (v1:rule:mod:entity)
    assert "TA001" in fps["stableFingerprint"]


def test_sarif_ordering_and_indices(mock_result):
    """
    SARIF uses a side-table for rules. We must ensure:
    1. Rules are sorted in driver.rules
    2. Results point to the correct index (ruleIndex)
    """
    stream = io.StringIO()
    SARIFReporter(dest=stream).report(mock_result)
    stream.seek(0)
    sarif = json.load(stream)

    run = sarif["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    results = run["results"]

    # 1. Verify Rule sorting
    assert rules[0]["id"] == "TA001"
    assert rules[1]["id"] == "TA002"

    # 2. Verify Index Mapping
    # Result for TA001 should point to index 0
    res_ta001 = next(r for r in results if r["ruleId"] == "TA001")
    assert res_ta001["ruleIndex"] == 0
    assert res_ta001["level"] == "error"

    # Result for TA002 should point to index 1
    res_ta002 = next(r for r in results if r["ruleId"] == "TA002")
    assert res_ta002["ruleIndex"] == 1
    assert res_ta002["level"] == "note"  # INFO -> note
