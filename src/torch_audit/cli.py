import importlib
import traceback

import click
import torch

from .__about__ import __version__
from .api import audit
from .core import Phase, Rule
from .registry import RuleRegistry
from .reporters.console import ConsoleReporter
from .reporters.json import JSONReporter
from .reporters.sarif import SARIFReporter


def load_model_from_string(import_str: str) -> torch.nn.Module:
    """Load a ``torch.nn.Module`` from an import string.

    The import string format is:

        ``module.path:ObjName``

    Where ``ObjName`` can be either:
      - a ``nn.Module`` instance
      - a ``nn.Module`` class (zero-arg constructor)

    Examples:
      - ``torchvision.models:resnet18``
      - ``my_project.models:MyModel``
      - ``my_project.models:MODEL``  # pre-instantiated
    """

    try:
        module_path, obj_name = import_str.split(":")
    except ValueError:
        raise click.BadParameter(
            f"Invalid format '{import_str}'. Use 'module.path:ModelClass' or 'module.path:model_instance'"
        ) from None

    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise click.BadParameter(
            f"Could not load '{obj_name}' from '{module_path}': {e}"
        ) from None

    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        try:
            return obj()
        except Exception as e:
            raise click.BadParameter(
                f"Could not instantiate model class '{obj_name}': {e}"
            ) from None

    if isinstance(obj, torch.nn.Module):
        return obj

    raise click.BadParameter(f"Object '{obj_name}' is not a torch.nn.Module")


def _parse_rule_list(
    _ctx: click.Context, _param: click.Parameter, value: tuple[str, ...]
) -> set[str]:
    """Parse repeated --select/--ignore values.

    Supports:
      - ``--select TA100 --select TA200``
      - ``--select TA100,TA200``
    """
    if not value:
        return set()

    rules: set[str] = set()
    for item in value:
        for token in item.split(","):
            t = token.strip()
            if t:
                rules.add(t)
    return rules


def _ensure_rules_loaded() -> list[Rule]:
    """Ensure all built-in rule modules are imported and registered."""
    # Importing validators registers rules in the global registry.
    from .loader import load_runtime_validators

    load_runtime_validators()
    return RuleRegistry.all_rules()


def _print_rules_table(rules: list[Rule]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="torch-audit rules", show_lines=False)
    table.add_column("ID", style="bold")
    table.add_column("Severity")
    table.add_column("Category")
    table.add_column("Title")

    for r in rules:
        table.add_row(r.id, r.default_severity.value, r.category, r.title)

    Console().print(table)


def _print_rule_detail(rule: Rule) -> None:
    from rich.console import Console
    from rich.panel import Panel

    body = (
        f"[bold]{rule.id}[/bold]  ({rule.category})\n"
        f"Default severity: [bold]{rule.default_severity.value}[/bold]\n\n"
        f"[bold]Description[/bold]\n{rule.description}\n\n"
        f"[bold]Remediation[/bold]\n{rule.remediation}"
    )

    Console().print(Panel.fit(body, title="Rule", border_style="cyan"))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="torch-audit")
@click.argument("target", required=False)
@click.option(
    "--list-rules",
    is_flag=True,
    help="List all available rule IDs and exit.",
)
@click.option(
    "--explain",
    metavar="RULE_ID",
    help="Show details for a single rule ID and exit.",
)
@click.option(
    "--format",
    "formats",
    type=click.Choice(["rich", "json", "sarif"], case_sensitive=False),
    multiple=True,
    default=["rich"],
    help="Output format(s). Can be specified multiple times. Default: rich.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(writable=True),
    help="Output file for machine-readable formats (JSON/SARIF).",
)
@click.option(
    "--fail-level",
    type=click.Choice(["INFO", "WARN", "ERROR", "FATAL"], case_sensitive=False),
    default="ERROR",
    help="Exit with non-zero status if findings meet this severity.",
)
@click.option(
    "--phase",
    type=click.Choice([Phase.STATIC.value, Phase.INIT.value], case_sensitive=False),
    default=Phase.STATIC.value,
    help="Context phase for the audit (e.g. static, init).",
)
@click.option("--step", type=int, default=0, help="The training step to simulate.")
@click.option(
    "--baseline",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Path to a baseline JSON file. Only new findings will trigger failure.",
)
@click.option(
    "--update-baseline",
    is_flag=True,
    help="Overwrite the baseline file with the findings from this run.",
)
@click.option(
    "--select",
    multiple=True,
    callback=_parse_rule_list,
    help="Only run specific rules (by ID). Can be comma-separated.",
)
@click.option(
    "--ignore",
    multiple=True,
    callback=_parse_rule_list,
    help="Ignore specific rules (by ID). Can be comma-separated.",
)
@click.option(
    "--show-suppressed",
    is_flag=True,
    help="Include suppressed findings in the output.",
)
@click.option(
    "--ignore-internal-errors",
    is_flag=True,
    help="Suppress internal validator crashes (TA000).",
)
def main(
    target: str | None,
    list_rules: bool,
    explain: str | None,
    formats: tuple[str, ...],
    output_path: str | None,
    fail_level: str,
    phase: str,
    step: int,
    baseline: str | None,
    update_baseline: bool,
    select: set[str],
    ignore: set[str],
    show_suppressed: bool,
    ignore_internal_errors: bool,
) -> None:
    """Audit a PyTorch model for silent training bugs.

    TARGET is an import string, for example:

      - ``torchvision.models:resnet18``
      - ``my_project.models:MyModel``

    Tip: if the ``torch-audit`` console script isn't available, run:

      ``python -m torch_audit ...``
    """

    # --- Rules-only modes ---
    if list_rules:
        rules = _ensure_rules_loaded()
        _print_rules_table(rules)
        raise SystemExit(0)

    if explain:
        _ensure_rules_loaded()
        rule = RuleRegistry.get(explain.strip())
        if rule is None:
            click.echo(f"Unknown rule id: {explain}", err=True)
            raise SystemExit(2)
        _print_rule_detail(rule)
        raise SystemExit(0)

    if not target:
        click.echo(click.get_current_context().get_help())
        raise SystemExit(2)

    # 1) Load user model
    click.secho(f"🔎 Loading target: {target}...", dim=True)
    try:
        model = load_model_from_string(target)
    except Exception as e:
        click.secho(f"FATAL: {e}", fg="red")
        raise SystemExit(1) from e

    # 2) Run audit (library API)
    try:
        result = audit(
            model=model,
            step=step,
            phase=phase,
            fail_level=fail_level,
            show_report=False,
            baseline_file=baseline,
            update_baseline=update_baseline,
            select_rules=select or None,
            ignore_rules=ignore or None,
            show_suppressed=bool(show_suppressed),
            suppress_internal_errors=bool(ignore_internal_errors),
        )

    except Exception as e:
        click.secho(f"Error running audit: {e}", err=True)
        traceback.print_exc()
        raise SystemExit(1) from None

    # 3) Report results
    requested_formats = {f.lower() for f in formats}

    file_formats = {"json", "sarif"}
    requested_file_formats = requested_formats.intersection(file_formats)
    if output_path and len(requested_file_formats) > 1:
        click.echo(
            "Error: --output cannot be used with multiple file formats (JSON + SARIF).",
            err=True,
        )
        raise SystemExit(2)

    if "rich" in requested_formats:
        ConsoleReporter().report(result)

    if "json" in requested_formats:
        JSONReporter(dest=output_path).report(result)

    if "sarif" in requested_formats:
        SARIFReporter(dest=output_path).report(result)

    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
