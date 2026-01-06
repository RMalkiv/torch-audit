import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List
from contextlib import contextmanager
from rich.console import Console
from rich.table import Table
from .config import AuditConfig

from ..modules.general import (
    hardware,
    optimization,
    hygiene,
    activations,
    graph,
    gradients
)

from ..modules.nlp import tokenization, structure
from ..modules.cv import images, layers

console = Console()


class Auditor:
    def __init__(
            self,
            model: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            config: Optional[Union[AuditConfig, Dict]] = None
    ):
        self._is_active_step = None
        self.model = model
        self.optimizer = optimizer

        if isinstance(config, dict):
            self.config = AuditConfig(**config)
        else:
            self.config = config or AuditConfig()

        self.issues = []

        self._step_count = 0
        self._audited_count = 0
        self._force_next_audit = False

        self._monitors = []

        if self.config.monitor_dead_neurons:
            self._monitors.append(activations.ActivationMonitor())

        if self.config.monitor_graph:
            self._monitors.append(graph.GraphMonitor())

        if self.config.monitor_nlp:
            self._monitors.append(tokenization.TokenizationMonitor(
                pad_token_id=self.config.pad_token_id,
                unk_token_id=self.config.unk_token_id,
                vocab_size=self.config.vocab_size
            ))


    def schedule_next_step(self):
        """
        Manually forces the next `audit_dynamic` call to run.
        Call this if you detect a loss spike or NaN in your training loop.
        """
        self._force_next_audit = True
        console.print("[bold yellow]⚠️  Audit scheduled for next step (Manual Trigger)[/bold yellow]")

    def audit_data(self, batch):
        """
        Checks input data hygiene (NaNs, shapes, ranges).
        Respects the scheduling interval unless forced.
        """
        should_audit = (self._step_count % self.config.interval == 0) or self._force_next_audit

        if not should_audit:
            return

        new_issues = []

        general_config = {
            "float_threshold": self.config.float_threshold,
            "check_batch_size": self.config.check_batch_size
        }
        new_issues.extend(hygiene.check_input_hygiene(batch, general_config))

        if self.config.monitor_cv:
            new_issues.extend(images.check_image_hygiene(batch))

        if new_issues:
            self.issues.extend(new_issues)
            self._print_report_fragment(new_issues)

    def start_dynamic_audit(self):
        """
        Manually starts the audit. Used by Callbacks (Lightning/HF) or custom loops.
        Returns True if audit started (scheduled), False otherwise.
        """
        is_scheduled = (self._step_count % self.config.interval == 0)

        if self.config.limit and self._audited_count >= self.config.limit:
            is_scheduled = False

        self._is_active_step = is_scheduled or self._force_next_audit

        if self._is_active_step:
            trigger_type = "Manual Trigger" if self._force_next_audit else f"Step {self._step_count}"
            console.print(f"\n[dim cyan]🚀 Audit Running ({trigger_type})...[/dim cyan]")

            for monitor in self._monitors:
                monitor.attach(self.model)

        return self._is_active_step

    def stop_dynamic_audit(self):
        """
        Manually stops the audit, cleans up hooks, and prints the report.
        Must be called if start_dynamic_audit() was called.
        """
        if self._is_active_step:
            try:
                for monitor in self._monitors:
                    monitor.detach()

                self._collect_dynamic_issues()
                self.show_report()

                self._audited_count += 1
                self._force_next_audit = False
            except Exception as e:
                console.print(f"[bold red]❌ Audit Reporting Failed:[/bold red] {e}")
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
            if started:
                console.print(f"[bold red]❌ Audit Failed with Exception:[/bold red] {e}")
            raise e
        finally:
            self.stop_dynamic_audit()

    def _collect_dynamic_issues(self):
        """Gather results from all active monitors and gradient checks."""
        for monitor in self._monitors:
            if hasattr(monitor, 'get_issues'):
                self.issues.extend(monitor.get_issues(self.model))

        grad_issues = gradients.check_gradients(self.model)
        self.issues.extend(grad_issues)

    def audit_static(self):
        """
        Runs one-time static analysis on model architecture and weights.
        """
        console.print(f"[bold cyan]🔍 Starting Static Audit on {self.model.__class__.__name__}...[/bold cyan]")

        self.issues.extend(hardware.check_tensor_core_alignment(self.model))

        if self.optimizer:
            self.issues.extend(optimization.check_weight_decay(self.model, self.optimizer))

        if self.config.monitor_nlp:
            self.issues.extend(structure.check_structure(
                self.model,
                vocab_size=self.config.vocab_size,
                pad_token_id=self.config.pad_token_id
            ))

        if self.config.monitor_cv:
            self.issues.extend(layers.check_conv_layers(self.model))

        self.show_report()

    def show_report(self):
        """Renders the issue table to the console."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        if not self.issues:
            console.print("[dim green]✅ Clean step[/dim green]")
            return

        self.issues.sort(key=lambda x: 0 if x.get('severity') == 'ERROR' else 1)

        table = Table(title=f"⚠️ Audit Report (Step {self._step_count})")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Layer", style="magenta")
        table.add_column("Message", style="white")

        for issue in self.issues:
            severity_icon = "🔴" if issue.get('severity') == "ERROR" else "🟡"
            table.add_row(
                f"{severity_icon} {issue['type']}",
                str(issue['layer']),
                issue['message']
            )

        console.print(table)
        self.issues = []

    def _print_report_fragment(self, specific_issues):
        """Helper to print immediate feedback (e.g. from audit_data)."""
        for issue in specific_issues:
            icon = "🔴" if issue['severity'] == "ERROR" else "🟡"
            console.print(f"   {icon} {issue['message']} (in {issue['layer']})")
