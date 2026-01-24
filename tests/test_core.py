from torch_audit.core import Finding, Phase, Rule, Severity


def test_severity_ordering():
    """
    Verifies that Severity levels are comparable and ordered correctly:
    INFO < WARN < ERROR < FATAL
    """
    # 1. Check absolute ordering
    assert Severity.INFO < Severity.WARN
    assert Severity.WARN < Severity.ERROR
    assert Severity.ERROR < Severity.FATAL

    # 2. Check transitive property
    assert Severity.INFO < Severity.FATAL

    # 3. Check equality and >= logic (critical for is_failure checks)
    assert Severity.ERROR >= Severity.WARN
    assert Severity.ERROR >= Severity.ERROR
    assert not (Severity.WARN >= Severity.ERROR)

    # 4. Sorting verification
    levels = [Severity.FATAL, Severity.INFO, Severity.ERROR, Severity.WARN]
    sorted_levels = sorted(levels)
    assert sorted_levels == [
        Severity.INFO,
        Severity.WARN,
        Severity.ERROR,
        Severity.FATAL,
    ]


def test_finding_fingerprint_stability():
    """
    Fingerprint must depend ONLY on (rule_id, module_path, entity).
    It must ignore volatile fields like step, phase, and metadata.
    """
    f1 = Finding(
        rule_id="TA001",
        message="Message A",
        severity=Severity.ERROR,
        module_path="layer1.weight",
        entity="grad",
        step=10,
        phase=Phase.BACKWARD,
        metadata={"loss": 0.5},
    )

    f2 = Finding(
        rule_id="TA001",
        message="Message B (Different)",
        severity=Severity.WARN,
        module_path="layer1.weight",
        entity="grad",
        step=500,
        phase=Phase.OPTIMIZER,
        metadata={"loss": 12.0},
    )

    assert f1.get_fingerprint() == f2.get_fingerprint()


def test_finding_fingerprint_sensitivity():
    """
    Fingerprint must change if rule, module, or entity changes.
    """
    base = Finding(
        rule_id="TA001",
        message="msg",
        severity=Severity.INFO,
        module_path="layer1",
        entity="weight",
    )

    diff_rule = Finding(
        rule_id="TA002",
        message="msg",
        severity=Severity.INFO,
        module_path="layer1",
        entity="weight",
    )
    assert base.get_fingerprint() != diff_rule.get_fingerprint()

    diff_mod = Finding(
        rule_id="TA001",
        message="msg",
        severity=Severity.INFO,
        module_path="layer2",
        entity="weight",
    )
    assert base.get_fingerprint() != diff_mod.get_fingerprint()

    diff_ent = Finding(
        rule_id="TA001",
        message="msg",
        severity=Severity.INFO,
        module_path="layer1",
        entity="bias",
    )
    assert base.get_fingerprint() != diff_ent.get_fingerprint()


def test_rule_serialization():
    """
    Rule.to_dict() should convert the Severity enum to a string value.
    """
    rule = Rule(
        id="TA100",
        title="Test Rule",
        description="Desc",
        remediation="Fix it",
        category="Test",
        default_severity=Severity.ERROR,
    )

    data = rule.to_dict()

    assert data["id"] == "TA100"
    assert data["default_severity"] == "ERROR"
    assert isinstance(data["default_severity"], str)


def test_finding_serialization():
    """
    Finding.to_dict() should recursively convert Enums (Severity, Phase) to strings.
    """
    finding = Finding(
        rule_id="TA100",
        message="Something wrong",
        severity=Severity.WARN,
        phase=Phase.BACKWARD,
        module_path="model.layer",
        step=1,
    )

    data = finding.to_dict()

    assert data["severity"] == "WARN"
    assert data["phase"] == "backward"

    assert not isinstance(data["severity"], Severity)
    assert not isinstance(data["phase"], Phase)
