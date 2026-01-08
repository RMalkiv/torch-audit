import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List, Any
from contextlib import contextmanager

from .config import AuditConfig
from .issue import AuditIssue
from .validator import Validator
from .reporter import Reporter, RichConsoleReporter

from ..modules.general.hardware import (
    TensorCoreValidator,
    MemoryLayoutValidator,
    PrecisionValidator,
    DevicePlacementValidator
)
from ..modules.general.hygiene import InputHygieneValidator
from ..modules.general.optimizer_config import OptimizerConfigValidator
from ..modules.general.stability import StabilityValidator
from ..modules.general.activations import ActivationValidator
from ..modules.general.graph import GraphValidator
from ..modules.general.gradients import GradientValidator

from ..modules.nlp.structure import StructureValidator
from ..modules.nlp.tokenization import TokenizationValidator
from ..modules.cv.layers import ConvValidator
from ..modules.cv.images import ImageBatchValidator


class Auditor:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            config: Optional[Union[AuditConfig, Dict]] = None,
            reporters: Optional[List[Reporter]] = None
    ):
        self._is_active_step = False
        self.model = model
        self.optimizer = optimizer

        if isinstance(config, dict):
            self.config = AuditConfig(**config)
        else:
            self.config = config or AuditConfig()

        # --- Reporters Setup ---
        if reporters is None:
            self.reporters = [RichConsoleReporter()]
        else:
            self.reporters = reporters

        self.issues: List[AuditIssue] = []
        self._step_count = 0
        self._audited_count = 0
        self._force_next_audit = False

        # --- Initialize Validators ---
        self.validators: List[Validator] = []

        # 1. Hardware & System Validators (Always Active)
        self.validators.extend([
            TensorCoreValidator(),
            MemoryLayoutValidator(),
            PrecisionValidator(),
            DevicePlacementValidator()
        ])

        # 2. Optimization Validators
        if self.optimizer:
            self.validators.append(OptimizerConfigValidator(self.optimizer))
            self.validators.append(StabilityValidator(self.optimizer))

        # 3. Data Hygiene
        self.validators.append(InputHygieneValidator(
            float_threshold=self.config.float_threshold,
            check_batch_size=self.config.check_batch_size
        ))

        # 4. Dynamic Monitors (Configurable)
        if self.config.monitor_dead_neurons:
            self.validators.append(ActivationValidator())

        if self.config.monitor_graph:
            atomic_modules = getattr(self.config, 'graph_atomic_modules', [])
            self.validators.append(GraphValidator(extra_atomic_modules=atomic_modules))

        self.validators.append(GradientValidator())

        # 5. Domain Specific Validators (NLP)
        if self.config.monitor_nlp:
            self.validators.append(StructureValidator(
                vocab_size=self.config.vocab_size,
                pad_token_id=self.config.pad_token_id
            ))
            self.validators.append(TokenizationValidator(
                vocab_size=self.config.vocab_size,
                pad_token_id=self.config.pad_token_id,
                unk_token_id=self.config.unk_token_id
            ))

        # 6. Domain Specific Validators (CV)
        if self.config.monitor_cv:
            self.validators.append(ConvValidator())
            self.validators.append(ImageBatchValidator())

    def schedule_next_step(self):
        """
        Manually forces the next `audit_dynamic` call to run.
        Call this if you detect a loss spike or NaN in your training loop.
        """
        self._force_next_audit = True
        if self.reporters and isinstance(self.reporters[0], RichConsoleReporter):
            self.reporters[0].console.print(
                "[bold yellow]⚠️  Audit scheduled for next step (Manual Trigger)[/bold yellow]")

    def audit_static(self):
        """
        Runs one-time static analysis on model architecture and weights.
        """
        if self.reporters and isinstance(self.reporters[0], RichConsoleReporter):
            self.reporters[0].console.print(
                f"[bold cyan]🔍 Starting Static Audit on {self.model.__class__.__name__}...[/bold cyan]")

        for validator in self.validators:
            self.issues.extend(validator.check_static(self.model))

        self.show_report()

    def audit_data(self, batch: Any):
        """
        Checks input data using all registered validators.
        """
        should_audit = (self._step_count % self.config.interval == 0) or self._force_next_audit

        if not should_audit:
            return

        new_issues = []
        for validator in self.validators:
            new_issues.extend(validator.check_data(batch))

        if new_issues:
            self.issues.extend(new_issues)
            self._print_report_fragment(new_issues)

    def start_dynamic_audit(self):
        """
        Attaches hooks for the dynamic audit.
        Returns True if audit started (hooks attached).
        """
        is_scheduled = (self._step_count % self.config.interval == 0)

        if self.config.limit and self._audited_count >= self.config.limit:
            is_scheduled = False

        self._is_active_step = is_scheduled or self._force_next_audit

        if self._is_active_step:
            if self.reporters and isinstance(self.reporters[0], RichConsoleReporter):
                trigger_type = "Manual Trigger" if self._force_next_audit else f"Step {self._step_count}"
                self.reporters[0].console.print(f"\n[dim cyan]🚀 Audit Running ({trigger_type})...[/dim cyan]")

            for validator in self.validators:
                validator.attach(self.model)

        return self._is_active_step

    def stop_dynamic_audit(self):
        """
        Detaches hooks, collects dynamic issues, and reports results.
        """
        if self._is_active_step:
            try:
                # 1. Detach hooks first to stop overhead
                for validator in self.validators:
                    validator.detach()

                # 2. Collect issues from the execution
                self._collect_dynamic_issues()

                # 3. Report
                self.show_report()

                self._audited_count += 1
                self._force_next_audit = False
            except Exception as e:
                # Log critical failure
                if self.reporters and isinstance(self.reporters[0], RichConsoleReporter):
                    self.reporters[0].console.print(f"[bold red]❌ Audit Reporting Failed:[/bold red] {e}")

                # Ensure detach happens even if reporting fails
                for validator in self.validators:
                    validator.detach()
            finally:
                self._is_active_step = False

        self._step_count += 1

    @contextmanager
    def audit_dynamic(self):
        """
        Context Manager that wraps start/stop logic.
        """
        started = self.start_dynamic_audit()
        try:
            yield
        except Exception as e:
            if started and self.reporters and isinstance(self.reporters[0], RichConsoleReporter):
                self.reporters[0].console.print(f"[bold red]❌ Audit Failed with Exception:[/bold red] {e}")
            raise e
        finally:
            self.stop_dynamic_audit()

    def audit_step(self, func):
        """
        Decorator to automatically wrap a training step function.
        Usage:
            @auditor.audit_step
            def training_step(batch): ...
        """

        def wrapper(*args, **kwargs):
            if args:
                candidate = args[0]
                if not isinstance(candidate, (nn.Module, Auditor)):
                    self.audit_data(candidate)
                elif len(args) > 1:
                    self.audit_data(args[1])

            with self.audit_dynamic():
                return func(*args, **kwargs)

        return wrapper

    def _collect_dynamic_issues(self):
        """
        Asks every validator if they found anything during the forward/backward pass.
        """
        for validator in self.validators:
            self.issues.extend(validator.check_dynamic(self.model))

    def show_report(self):
        """
        Delegates reporting to all attached reporters.
        """
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        for rep in self.reporters:
            rep.report(self._step_count, self.issues)

        self.issues = []

    def _print_report_fragment(self, specific_issues: List[AuditIssue]):
        """
        Helper to print immediate feedback for data errors to the console
        (if RichConsoleReporter is available).
        """
        if not self.reporters or not isinstance(self.reporters[0], RichConsoleReporter):
            return

        console = self.reporters[0].console
        specific_issues.sort()

        for issue in specific_issues:
            icon = "🔴" if issue.severity == "ERROR" else "🟡"
            console.print(f"   {icon} {issue.message} (in {issue.layer})")
