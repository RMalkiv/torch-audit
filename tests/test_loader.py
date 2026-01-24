from torch_audit.loader import load_default_validators
from torch_audit.validator import BaseValidator


def test_load_default_validators():
    validators = load_default_validators()

    assert len(validators) > 0
    assert isinstance(validators[0], BaseValidator)

    from torch_audit.validators.builtin.stability import StabilityValidator

    assert any(isinstance(v, StabilityValidator) for v in validators)
