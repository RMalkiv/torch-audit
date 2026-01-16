from typing import List, Type

from .validator import BaseValidator


def load_default_validators() -> List[BaseValidator]:
    """
    Instantiates and returns the standard set of validators.
    """
    # 1. Local Imports
    from .validators.builtin.architecture import ArchitectureValidator
    from .validators.builtin.data import DataValidator
    from .validators.builtin.hardware import HardwareValidator
    from .validators.builtin.optimization import OptimizerValidator
    from .validators.builtin.stability import StabilityValidator

    # 2. Define the Default Registry
    DEFAULT_CLASSES: List[Type[BaseValidator]] = [
        StabilityValidator,
        ArchitectureValidator,
        OptimizerValidator,
        HardwareValidator,
        DataValidator,
    ]

    # 3. Instantiate
    return [cls() for cls in DEFAULT_CLASSES]


def load_runtime_validators() -> List[BaseValidator]:
    """Load validators suitable for **runtime** / training-loop audits.

    This includes the default stateless validators plus stateful, hook-based
    validators (e.g., Graph / Activation) that require attach/detach.
    """

    # Start with the standard validators.
    validators: List[BaseValidator] = load_default_validators()

    # Add runtime/stateful validators.
    from .validators.builtin.activation import ActivationValidator
    from .validators.builtin.graph import GraphValidator

    validators.extend(
        [
            GraphValidator(),
            ActivationValidator(),
        ]
    )
    return validators
