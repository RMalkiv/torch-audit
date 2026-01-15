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
