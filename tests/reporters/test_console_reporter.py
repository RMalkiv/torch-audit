import io
import re

import pytest
from rich.console import Console

from torch_audit.core import AuditResult, Finding, Rule, Severity
from torch_audit.reporters.console import ConsoleReporter


def strip_ansi(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture
def mock_result():
    r1 = Rule("TA001", "ErrRule", "Desc", "Fix", "Cat", Severity.ERROR)
    f1 = Finding("TA001", "Something error", Severity.ERROR, module_path="layer1")

    return AuditResult(
        findings=[f1],
        exit_code=1,
        max_severity=Severity.ERROR,
        suppressed_count=5,
        rules={"TA001": r1},
    )


def test_console_reporter_output(mock_result):
    buf = io.StringIO()
    reporter = ConsoleReporter()
    reporter.console = Console(file=buf, force_terminal=True, width=120)

    reporter.report(mock_result)

    raw_output = buf.getvalue()
    clean_output = strip_ansi(raw_output)

    # 1. Check Header
    assert "Audit Finished" in clean_output
    assert "Errors: 1" in clean_output
    assert "Suppressed: 5" in clean_output

    # 2. Check Table
    assert "TA001: ErrRule" in clean_output
    assert "layer1" in clean_output
    assert "Something error" in clean_output

    # 3. Check Footer
    assert "Audit Failed" in clean_output
