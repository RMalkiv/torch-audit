from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from .config import AuditConfig
from .context import AuditContext, AuditState
from .core import AuditResult, Phase, Severity
from .loader import load_runtime_validators
from .reporters.base import Reporter
from .runner import AuditRunner
from .validator import BaseValidator


def _normalize_severity(value: Union[str, Severity], *, strict: bool) -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity(str(value).upper())
    except ValueError:
        if strict:
            raise ValueError(
                f"Invalid fail_level: '{value}'. Must be one of {[s.value for s in Severity]}"
            ) from None
        return Severity.ERROR


def _normalize_phase(value: Union[str, Phase], *, strict: bool) -> Phase:
    if isinstance(value, Phase):
        return value
    try:
        return Phase(str(value).lower())
    except ValueError:
        if strict:
            raise ValueError(
                f"Invalid phase: '{value}'. Must be one of {[p.value for p in Phase]}"
            ) from None
        return Phase.STATIC


def _pack_batch(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Best-effort packing of model inputs into a batch object.

    We keep this intentionally simple so :meth:`AuditContext.iter_batch_tensors`
    can still discover tensors inside nested structures.
    """
    if len(args) == 1 and not kwargs:
        return args[0]
    if len(args) == 0 and kwargs:
        return kwargs
    # Mixed args/kwargs: expose both.
    return {"args": list(args), "kwargs": kwargs}


class Auditor:
    """A training-loop friendly auditor.

    The auditor holds an :class:`~torch_audit.runner.AuditRunner` and manages
    optional hook lifecycle for validators that implement :meth:`BaseValidator.attach`.

    Typical usage:

    ```python
    auditor = Auditor(model, optimizer=opt)
    with auditor:
        auditor.audit_static()
        auditor.audit_init()
        out = auditor.forward(batch)
        loss = ...
        auditor.backward(loss)
        auditor.optimizer_step()
    result = auditor.finish()
    ```
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        start_step: int = 0,
        every_n_steps: int = 1,
        validators: Optional[List[BaseValidator]] = None,
        reporters: Optional[List[Reporter]] = None,
        fail_level: Union[str, Severity] = Severity.ERROR,
        strict: bool = False,
        baseline_file: Optional[str] = None,
        update_baseline: bool = False,
        select_rules: Optional[Set[str]] = None,
        ignore_rules: Optional[Set[str]] = None,
        show_suppressed: bool = False,
        suppress_internal_errors: bool = False,
        auto_finish: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.step = int(start_step)
        self.every_n_steps = max(1, int(every_n_steps))
        self._strict = bool(strict)
        self._auto_finish = bool(auto_finish)

        self._fail_level = _normalize_severity(fail_level, strict=self._strict)

        self.config = AuditConfig(
            fail_level=self._fail_level,
            baseline_file=baseline_file,
            update_baseline=update_baseline,
            select_rules=select_rules,
            ignore_rules=ignore_rules,
            show_suppressed=show_suppressed,
            suppress_internal_errors=suppress_internal_errors,
        )

        self.validators: List[BaseValidator] = (
            list(validators) if validators is not None else load_runtime_validators()
        )

        self.runner = AuditRunner(self.config, self.validators)
        self.reporters: List[Reporter] = list(reporters) if reporters else []

        self._attached: bool = False
        self._attach_depth: int = 0
        self._finished: bool = False
        self._final_result: Optional[AuditResult] = None

        self._last_batch: Any = None
        self._data_audited_this_step: bool = False

        # Used for "delta" retrieval.
        self._findings_cursor: int = 0

    # --- Context manager ---

    def __enter__(self) -> "Auditor":
        self._attach_depth += 1
        if not self._attached:
            self.attach()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._auto_finish and not self._finished:
                self.finish()
        finally:
            self._attach_depth = max(0, self._attach_depth - 1)
            if self._attached and self._attach_depth == 0:
                self.detach()

    # --- Lifecycle ---

    def attach(self) -> None:
        """Attach hook-based validators to the model."""
        if self._attached:
            return
        for v in self.validators:
            v.attach(self.model)
        self._attached = True

    def detach(self) -> None:
        """Detach hook-based validators from the model."""
        if not self._attached:
            return
        # Detach in reverse order as a mild safety measure.
        for v in reversed(self.validators):
            try:
                v.detach()
            except Exception:
                # Keep detach best-effort: we don't want cleanup failures to crash training.
                pass
        self._attached = False

    # --- Public accessors ---

    @property
    def findings(self):
        return self.runner.findings

    def pop_new_findings(self):
        """Return findings produced since the last call.

        Useful for streaming console output in training loops.
        """
        new = self.runner.findings[self._findings_cursor :]
        self._findings_cursor = len(self.runner.findings)
        return new

    # --- Core execution helpers ---

    def _should_audit(self, step: int, phase: Phase) -> bool:
        # Always allow static/init audits (usually one-shot).
        if phase in {Phase.STATIC, Phase.INIT}:
            return True
        return (step % self.every_n_steps) == 0

    def _make_context(
        self, phase: Phase, *, step: Optional[int], batch: Any
    ) -> AuditContext:
        st = AuditState(
            model=self.model,
            step=self.step if step is None else int(step),
            phase=phase,
            batch=batch,
            optimizer=self.optimizer,
        )
        return AuditContext(st)

    def _call_phase_start(self, ctx: AuditContext) -> None:
        for v in self.validators:
            v.on_phase_start(ctx)

    def _call_phase_end(self, ctx: AuditContext) -> None:
        for v in self.validators:
            v.on_phase_end(ctx)

    # --- High-level phase APIs ---

    def audit_static(self, *, step: Optional[int] = None) -> None:
        """Run static validators (architecture/hardware/etc)."""
        ctx = self._make_context(Phase.STATIC, step=step, batch=None)
        if self._should_audit(ctx.step, ctx.phase):
            self.runner.run_step(ctx)

    def audit_init(self, *, step: Optional[int] = None) -> None:
        """Run init validators (optimizer config, hardware, etc)."""
        ctx = self._make_context(Phase.INIT, step=step, batch=None)
        if self._should_audit(ctx.step, ctx.phase):
            self.runner.run_step(ctx)

    def audit_data(self, batch: Any, *, step: Optional[int] = None) -> None:
        """Run data-only checks on a batch.

        This intentionally runs only data validators to avoid false positives from
        hook-based forward validators (Graph/Activation) before a forward pass.
        """

        self._last_batch = batch
        self._data_audited_this_step = True

        ctx = self._make_context(Phase.FORWARD, step=step, batch=batch)
        if not self._should_audit(ctx.step, ctx.phase):
            return

        # Filter to data validators (best-effort).
        try:
            from .validators.builtin.data import DataValidator

            data_validators = [
                v for v in self.validators if isinstance(v, DataValidator)
            ]
        except Exception:
            data_validators = []

        if not data_validators:
            return

        self.runner.run_step(ctx, validators=data_validators)

    # --- Wrapped training helpers ---

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run a model forward pass and then audit the FORWARD phase.

        This wrapper is the easiest way to ensure stateful validators are run
        at the correct time (after the forward hooks have collected data).
        """

        if not self._attached:
            self.attach()

        batch = _pack_batch(args, kwargs)
        self._last_batch = batch
        self._data_audited_this_step = False

        # Let validators reset per-forward state.
        pre_ctx = self._make_context(Phase.FORWARD, step=None, batch=batch)
        self._call_phase_start(pre_ctx)

        # Optionally run data checks *before* forward to catch device mismatches early.
        if self._should_audit(pre_ctx.step, pre_ctx.phase):
            self.audit_data(batch)

        # Forward pass
        out = self.model(*args, **kwargs)

        # Post-forward audit (exclude data validators to avoid duplicate findings).
        post_ctx = self._make_context(Phase.FORWARD, step=None, batch=batch)

        if self._should_audit(post_ctx.step, post_ctx.phase):
            # Exclude data validators because we already ran them pre-forward.
            try:
                from .validators.builtin.data import DataValidator

                post_validators = [
                    v for v in self.validators if not isinstance(v, DataValidator)
                ]
            except Exception:
                post_validators = self.validators

            self.runner.run_step(post_ctx, validators=post_validators)

        self._call_phase_end(post_ctx)
        return out

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Run backward and then audit the BACKWARD phase."""

        if not self._attached:
            self.attach()

        ctx = self._make_context(Phase.BACKWARD, step=None, batch=self._last_batch)
        self._call_phase_start(ctx)

        loss.backward(*args, **kwargs)

        if self._should_audit(ctx.step, ctx.phase):
            self.runner.run_step(ctx)

        self._call_phase_end(ctx)

    def optimizer_step(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run optimizer.step() and then audit the OPTIMIZER phase.

        The auditor's internal step counter is incremented after the optimizer step.
        """

        if optimizer is not None:
            self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("Auditor.optimizer_step() requires an optimizer")

        if not self._attached:
            self.attach()

        ctx = self._make_context(Phase.OPTIMIZER, step=None, batch=self._last_batch)
        self._call_phase_start(ctx)

        self.optimizer.step(*args, **kwargs)

        if self._should_audit(ctx.step, ctx.phase):
            self.runner.run_step(ctx)

        self._call_phase_end(ctx)

        # Next training step.
        self.step += 1
        self._data_audited_this_step = False

    # --- Finalization ---

    def finish(self, *, report: bool = False) -> AuditResult:
        """Finalize and compute an :class:`~torch_audit.core.AuditResult`.

        This can be called multiple times; subsequent calls return the cached result.
        """
        if self._finished and self._final_result is not None:
            return self._final_result

        self._final_result = self.runner.finish()
        self._finished = True

        if report:
            for reporter in self.reporters:
                reporter.report(self._final_result)

        return self._final_result


@contextmanager
def audit_dynamic(
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    start_step: int = 0,
    every_n_steps: int = 1,
    validators: Optional[List[BaseValidator]] = None,
    reporters: Optional[List[Reporter]] = None,
    fail_level: Union[str, Severity] = Severity.ERROR,
    strict: bool = False,
    baseline_file: Optional[str] = None,
    update_baseline: bool = False,
    select_rules: Optional[Set[str]] = None,
    ignore_rules: Optional[Set[str]] = None,
    show_suppressed: bool = False,
    suppress_internal_errors: bool = False,
    run_static: bool = True,
    run_init: bool = True,
) -> Generator[Auditor, None, None]:
    """Convenience context manager for runtime auditing.

    On enter, attaches hook-based validators. Optionally runs static/init checks
    once at the start.
    """
    auditor = Auditor(
        model,
        optimizer=optimizer,
        start_step=start_step,
        every_n_steps=every_n_steps,
        validators=validators,
        reporters=reporters,
        fail_level=fail_level,
        strict=strict,
        baseline_file=baseline_file,
        update_baseline=update_baseline,
        select_rules=select_rules,
        ignore_rules=ignore_rules,
        show_suppressed=show_suppressed,
        suppress_internal_errors=suppress_internal_errors,
        auto_finish=False,
    )

    with auditor:
        if run_static:
            auditor.audit_static()
        if run_init:
            auditor.audit_init()
        yield auditor


def audit_step(
    auditor: Auditor,
    *,
    every_n_steps: Optional[int] = None,
    batch_extractor: Optional[Callable[..., Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to instrument a typical training-step function.

    This is intentionally lightweight ("opt-in") and does not attempt to
    introspect your training loop. The wrapped function receives no API changes;
    the decorator runs a **post-step** optimizer-phase audit by default.

    If you want precise phase audits, prefer the explicit `auditor.forward()`,
    `auditor.backward()`, and `auditor.optimizer_step()` wrappers.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if every_n_steps is not None:
                auditor.every_n_steps = max(1, int(every_n_steps))

            batch = None
            if batch_extractor is not None:
                try:
                    batch = batch_extractor(*args, **kwargs)
                except Exception:
                    batch = None

            # Run the step.
            out = fn(*args, **kwargs)

            # If we can, run a post-step audit (optimizer phase). This is best-effort.
            if auditor.optimizer is not None:
                ctx = auditor._make_context(Phase.OPTIMIZER, step=None, batch=batch)
                if auditor._should_audit(ctx.step, ctx.phase):
                    auditor.runner.run_step(ctx)

            return out

        return wrapped

    return decorator
