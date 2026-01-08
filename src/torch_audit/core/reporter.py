import logging
import json
from typing import List
from rich.console import Console
from rich.table import Table
from .issue import AuditIssue


class Reporter:
    def report(self, step_count: int, issues: List[AuditIssue]):
        raise NotImplementedError


class RichConsoleReporter(Reporter):
    """
    Renders beautiful tables to the console (Default).
    """

    def __init__(self):
        self.console = Console()

    def report(self, step_count: int, issues: List[AuditIssue]):
        if not issues:
            self.console.print("[dim green]✅ Clean step[/dim green]")
            return

        issues.sort()

        table = Table(title=f"⚠️ Audit Report (Step {step_count})")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Layer", style="magenta")
        table.add_column("Message", style="white")

        for issue in issues:
            icon = "🔴" if issue.severity == "ERROR" else "🟡"
            if issue.severity == "INFO": icon = "🔵"

            table.add_row(f"{icon} {issue.type}", str(issue.layer), issue.message)

        self.console.print(table)


class LogReporter(Reporter):
    """
    Writes issues to Python's standard logging facility.
    Great for production/headless training.
    """

    def __init__(self, logger_name: str = "torch_audit"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def report(self, step_count: int, issues: List[AuditIssue]):
        if not issues:
            return

        issues.sort()

        self.logger.info(f"--- Audit Report (Step {step_count}) ---")
        for issue in issues:
            msg = f"[{issue.type}] {issue.message} (Layer: {issue.layer})"

            if issue.severity == "ERROR":
                self.logger.error(msg)
            elif issue.severity == "WARNING":
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
