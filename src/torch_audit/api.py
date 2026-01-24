from typing import Any

import torch

from .config import AuditConfig
from .context import AuditContext, AuditState
from .core import AuditResult, Phase, Severity
from .loader import load_default_validators
from .reporters.base import Reporter
from .runner import AuditRunner
from .validator import BaseValidator


def audit(
    model: torch.nn.Module,
    step: int = 0,
    phase: str | Phase = Phase.STATIC,
    fail_level: str | Severity = Severity.ERROR,
    batch: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    validators: list[BaseValidator] | None = None,
    reporters: list[Reporter] | None = None,
    show_report: bool = False,
    strict: bool = False,
    baseline_file: str | None = None,
    update_baseline: bool = False,
    # CLI / filtering controls
    select_rules: set[str] | None = None,
    ignore_rules: set[str] | None = None,
    show_suppressed: bool = False,
    suppress_internal_errors: bool = False,
) -> AuditResult:
    """
    Run a one-shot audit on the given model.
    """
    # 1. Normalize Inputs (Phase)
    if not isinstance(phase, Phase):
        try:
            phase = Phase(str(phase).lower())
        except ValueError:
            if strict:
                raise ValueError(
                    f"Invalid phase: '{phase}'. Must be one of {[p.value for p in Phase]}"
                ) from None
            phase = Phase.STATIC

    # 2. Normalize Inputs (Severity)
    if not isinstance(fail_level, Severity):
        try:
            fail_level = Severity(str(fail_level).upper())
        except ValueError:
            if strict:
                raise ValueError(
                    f"Invalid fail_level: '{fail_level}'. Must be one of {[s.value for s in Severity]}"
                ) from None
            fail_level = Severity.ERROR

    # 3. Setup Configuration
    config = AuditConfig(
        fail_level=fail_level,
        baseline_file=baseline_file,
        update_baseline=update_baseline,
        select_rules=select_rules,
        ignore_rules=ignore_rules,
        show_suppressed=show_suppressed,
        suppress_internal_errors=suppress_internal_errors,
    )

    # 4. Load Validators (Lazy)
    if validators is None:
        validators = load_default_validators()

    # 5. Initialize Runner
    runner = AuditRunner(config, validators)

    # 6. Build Context
    state = AuditState(
        model=model, step=step, phase=phase, batch=batch, optimizer=optimizer
    )
    context = AuditContext(state)

    # 7. Execute
    runner.run_step(context)
    result = runner.finish()

    # 8. Reporting
    active_reporters = list(reporters) if reporters else []

    if show_report:
        from .reporters.console import ConsoleReporter

        active_reporters.append(ConsoleReporter())

    for reporter in active_reporters:
        reporter.report(result)

    return result
