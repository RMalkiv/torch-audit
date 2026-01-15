import pytest

from torch_audit.core import Rule, Severity
from torch_audit.registry import RuleRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Automatically clear registry before/after each test."""
    RuleRegistry.clear()
    yield
    RuleRegistry.clear()


def test_register_rule():
    r1 = Rule("TEST001", "Title", "Desc", "Fix", "Cat", Severity.ERROR)
    RuleRegistry.register(r1)

    assert RuleRegistry.get("TEST001") == r1
    assert len(RuleRegistry.all_rules()) == 1


def test_register_duplicate_id_conflict():
    """Registering a DIFFERENT rule with SAME ID should raise ValueError."""
    r1 = Rule("TEST001", "Title 1", "Desc", "Fix", "Cat", Severity.ERROR)
    r2 = Rule("TEST001", "Title 2", "Desc", "Fix", "Cat", Severity.INFO)  # Conflict

    RuleRegistry.register(r1)
    with pytest.raises(ValueError, match="Rule ID conflict"):
        RuleRegistry.register(r2)


def test_register_duplicate_id_idempotent():
    """Registering the EXACT SAME rule object twice is fine (idempotent)."""
    r1 = Rule("TEST001", "Title", "Desc", "Fix", "Cat", Severity.ERROR)

    RuleRegistry.register(r1)
    RuleRegistry.register(r1)

    assert len(RuleRegistry.all_rules()) == 1


def test_all_rules_sorting():
    """Rules should be returned sorted by ID."""
    r1 = Rule("B001", "B", "D", "F", "C", Severity.INFO)
    r2 = Rule("A001", "A", "D", "F", "C", Severity.INFO)

    RuleRegistry.register(r1)
    RuleRegistry.register(r2)

    rules = RuleRegistry.all_rules()
    assert rules[0].id == "A001"
    assert rules[1].id == "B001"
