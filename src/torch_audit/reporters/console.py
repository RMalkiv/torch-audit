from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core import AuditResult, Phase, Severity


class ConsoleReporter:
    def __init__(self):
        self.console = Console()
        self.phase_map = {
            Phase.STATIC: "STAT",
            Phase.INIT: "INIT",
            Phase.FORWARD: "FWD",
            Phase.BACKWARD: "BWD",
            Phase.OPTIMIZER: "OPT",
        }

    def report(self, result: AuditResult):
        if not result.findings:
            self.console.print(Panel("[green]✔ Audit Passed[/]", title="Torch Audit"))
            return

        # 1. Summary Header
        counts = {sev: 0 for sev in Severity}
        for f in result.findings:
            counts[f.severity] += 1

        # Build header string
        parts = []
        if counts[Severity.FATAL] > 0:
            parts.append(f"[bold red]Fatal: {counts[Severity.FATAL]}[/]")
        if counts[Severity.ERROR] > 0:
            parts.append(f"[red]Errors: {counts[Severity.ERROR]}[/]")
        if counts[Severity.WARN] > 0:
            parts.append(f"[yellow]Warn: {counts[Severity.WARN]}[/]")
        if counts[Severity.INFO] > 0:
            parts.append(f"[blue]Info: {counts[Severity.INFO]}[/]")

        if result.suppressed_count > 0:
            parts.append(f"[dim]Suppressed: {result.suppressed_count}[/]")

        header_text = ", ".join(parts)

        # Enhanced Title Logic
        if result.baseline_applied:
            title = f"Audit Finished (Max New: {result.max_severity.value} | Total: {result.max_severity_all.value})"
        else:
            title = f"Audit Finished (Max: {result.max_severity.value})"

        style = "red" if result.exit_code != 0 else "yellow"

        self.console.rule(f"[{style}]{title}[/]")
        self.console.print(f"[justify=center]{header_text}[/justify]")
        if result.baseline_applied:
            self.console.print(
                f"[justify=center][dim]Baseline Applied: {result.new_findings_count} new findings[/dim][/justify]"
            )
        self.console.print()

        # 2. Group & Sort
        grouped = {}
        for f in result.findings:
            grouped.setdefault(f.rule_id, []).append(f)

        sorted_rule_ids = sorted(grouped.keys())

        # 3. Render Tables
        for rule_id in sorted_rule_ids:
            findings = grouped[rule_id]
            # Deterministic sort
            findings.sort(
                key=lambda x: (x.module_path or "", x.entity or "", x.message)
            )

            rule = result.rules.get(rule_id)
            rule_title = rule.title if rule else "Unknown Rule"

            table = Table(
                title=f"[{style}]{rule_id}: {rule_title}", expand=True, show_lines=False
            )
            table.add_column("Sev", width=6, justify="center")
            table.add_column("Location")
            table.add_column("Ctx", style="dim", width=6)
            table.add_column("Message")

            for f in findings:
                # Color logic
                sev_color = "blue"  # Default for INFO
                if f.severity == Severity.WARN:
                    sev_color = "yellow"
                elif f.severity in [Severity.ERROR, Severity.FATAL]:
                    sev_color = "red"

                # Format Location
                loc = f"[bold]{f.module_path}[/]" if f.module_path else "global"
                if f.entity:
                    loc += f"\n[dim]{f.entity}[/]"

                # Format Context
                phase_str = self.phase_map.get(f.phase, f.phase.value[:4].upper())
                ctx = f"S:{f.step}\n{phase_str}" if f.step is not None else phase_str

                table.add_row(
                    f"[{sev_color}]{f.severity.value}[/]", loc, ctx, f.message
                )

            self.console.print(table)

            # Remediation Footer
            if rule and rule.remediation:
                self.console.print(
                    f"[dim italic]   ➜ Remediation: {rule.remediation}[/]"
                )
            self.console.print()

        # 4. Final Footer
        if result.exit_code != 0:
            self.console.print(
                "[red]Audit Failed.[/] [dim]Use --fail-level to adjust.[/dim]"
            )
