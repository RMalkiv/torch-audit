import json

from torch_audit.config import AuditConfig, Suppression
from torch_audit.core import Finding, Phase, Rule, Severity
from torch_audit.runner import INTERNAL_ERROR_RULE, AuditRunner
from torch_audit.validator import BaseValidator

# --- Helper: Mock Validator ---


class MockValidator(BaseValidator):
    def __init__(
        self,
        rule_id="MOCK001",
        should_crash=False,
        findings=None,
        supported_phases=None,
    ):
        self._rule = Rule(
            id=rule_id,
            title="Mock Rule",
            description="Desc",
            remediation="Fix",
            category="Test",
            default_severity=Severity.ERROR,
        )
        self.should_crash = should_crash
        self.findings = findings or []
        self._supported_phases = supported_phases
        self.check_called = False

    @property
    def rule(self):
        return self._rule

    @property
    def supported_phases(self):
        return self._supported_phases

    def check(self, context):
        self.check_called = True
        if self.should_crash:
            raise ValueError("Validator Crashed Boom!")

        yield from self.findings


# --- Tests ---


def test_runner_execution_basic(audit_context):
    """
    Runner should execute validators and collect findings correctly.
    """
    f1 = Finding("MOCK001", "msg", Severity.ERROR)
    validator = MockValidator(findings=[f1])

    config = AuditConfig(fail_level=Severity.ERROR)
    runner = AuditRunner(config, [validator])

    runner.run_step(audit_context)
    result = runner.finish()

    assert len(result.findings) == 1
    assert result.findings[0] == f1
    assert result.exit_code == 1  # ERROR matches fail_level


def test_runner_crash_handling(audit_context):
    """
    If a validator crashes, it should be caught and converted to a TA000 finding.
    """
    validator = MockValidator(should_crash=True)

    config = AuditConfig(suppress_internal_errors=False)
    runner = AuditRunner(config, [validator])

    runner.run_step(audit_context)
    result = runner.finish()

    assert len(result.findings) == 1
    crash_finding = result.findings[0]

    assert crash_finding.rule_id == INTERNAL_ERROR_RULE.id
    assert "Validator Crashed Boom!" in crash_finding.message
    assert "traceback" in crash_finding.metadata


def test_baseline_filtering(audit_context, tmp_path):
    """
    Baseline Logic:
    - Existing findings (in baseline) -> Do NOT trigger failure.
    - New findings -> DO trigger failure.
    """
    # 1. Setup Baseline File
    baseline_file = tmp_path / "baseline.json"

    finding_known = Finding(
        "MOCK001", "Known Issue", Severity.ERROR, module_path="layer1"
    )
    finding_new = Finding("MOCK001", "New Issue", Severity.ERROR, module_path="layer2")

    with open(baseline_file, "w") as f:
        json.dump([finding_known.get_fingerprint()], f)

    # 2. Configure Runner
    config = AuditConfig(
        fail_level=Severity.ERROR,
        baseline_file=str(baseline_file),
        update_baseline=False,
    )

    validator = MockValidator(findings=[finding_known, finding_new])
    runner = AuditRunner(config, [validator])

    # 3. Run
    runner.run_step(audit_context)
    result = runner.finish()

    # 4. Verify
    assert len(result.findings) == 2

    assert result.exit_code == 1


def test_baseline_filtering_pass(audit_context, tmp_path):
    """
    If all findings are in the baseline, exit_code should be 0 (Pass).
    """
    baseline_file = tmp_path / "baseline.json"
    finding_known = Finding("MOCK001", "Known Issue", Severity.ERROR)

    with open(baseline_file, "w") as f:
        json.dump([finding_known.get_fingerprint()], f)

    config = AuditConfig(baseline_file=str(baseline_file))
    validator = MockValidator(findings=[finding_known])
    runner = AuditRunner(config, [validator])

    runner.run_step(audit_context)
    result = runner.finish()

    assert len(result.findings) == 1
    assert result.exit_code == 0  # Pass!


def test_baseline_update(audit_context, tmp_path):
    """
    With update_baseline=True, runner should write ALL findings to the file.
    """
    baseline_file = tmp_path / "baseline.json"
    assert not baseline_file.exists()

    f1 = Finding("MOCK001", "Issue 1", Severity.ERROR)
    f2 = Finding("MOCK001", "Issue 2", Severity.WARN)

    config = AuditConfig(baseline_file=str(baseline_file), update_baseline=True)

    validator = MockValidator(findings=[f1, f2])
    runner = AuditRunner(config, [validator])

    runner.run_step(audit_context)
    runner.finish()

    assert baseline_file.exists()

    with open(baseline_file) as f:
        saved_fingerprints = json.load(f)

    assert len(saved_fingerprints) == 2
    assert f1.get_fingerprint() in saved_fingerprints
    assert f2.get_fingerprint() in saved_fingerprints


def test_runner_suppression_logic(audit_context):
    """
    Suppressed findings should not appear in results, but count should increment.
    """
    f1 = Finding("MOCK001", "msg", Severity.ERROR, module_path="ignored_layer")
    validator = MockValidator(findings=[f1])

    # Suppress MOCK001 on 'ignored_layer'
    suppression = Suppression(
        rule_id="MOCK001", reason="ignore", module_regex="ignored_layer"
    )
    config = AuditConfig(suppressions=[suppression])

    runner = AuditRunner(config, [validator])
    runner.run_step(audit_context)
    result = runner.finish()

    # Verify behavior
    assert len(result.findings) == 0
    assert result.suppressed_count == 1
    assert result.exit_code == 0  # Suppressed findings don't trigger failure


def test_runner_suppress_internal_errors(audit_context):
    """
    If suppress_internal_errors=True, validator crashes should be ignored (no TA000).
    """
    validator = MockValidator(should_crash=True)
    config = AuditConfig(suppress_internal_errors=True)

    runner = AuditRunner(config, [validator])
    runner.run_step(audit_context)
    result = runner.finish()

    assert len(result.findings) == 0
    assert result.exit_code == 0


def test_baseline_edge_cases(audit_context, tmp_path):
    """
    Baseline loading should fail gracefully (safe fallback) on IO/Data errors.
    """
    # 1. Missing File -> Safe (empty set)
    config_missing = AuditConfig(baseline_file=str(tmp_path / "missing.json"))
    runner = AuditRunner(config_missing, [])
    assert runner.baseline_fingerprints == set()

    # 2. Invalid JSON content -> Safe (empty set)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ not json }")
    config_bad = AuditConfig(baseline_file=str(bad_json))
    runner = AuditRunner(config_bad, [])
    assert runner.baseline_fingerprints == set()

    # 3. Wrong JSON Type (Dict instead of List) -> Safe (empty set)
    wrong_type = tmp_path / "wrong.json"
    with open(wrong_type, "w") as f:
        json.dump({"foo": "bar"}, f)

    config_wrong = AuditConfig(baseline_file=str(wrong_type))
    runner = AuditRunner(config_wrong, [])
    assert runner.baseline_fingerprints == set()


def test_runner_phase_filtering(audit_context):
    """
    Validators should only run if the context phase matches their supported_phases.
    """
    # audit_context fixture starts in Phase.STATIC

    # Case 1: Validator explicitly supports STATIC -> Should run
    v_static = MockValidator(supported_phases={Phase.STATIC})
    runner = AuditRunner(AuditConfig(), [v_static])
    runner.run_step(audit_context)
    assert v_static.check_called is True

    # Case 2: Validator only supports FORWARD -> Should NOT run in STATIC context
    v_forward = MockValidator(supported_phases={Phase.FORWARD})
    runner = AuditRunner(AuditConfig(), [v_forward])
    runner.run_step(audit_context)
    assert v_forward.check_called is False

    # Case 3: Validator has None (supports all) -> Should run
    v_all = MockValidator(supported_phases=None)
    runner = AuditRunner(AuditConfig(), [v_all])
    runner.run_step(audit_context)
    assert v_all.check_called is True
