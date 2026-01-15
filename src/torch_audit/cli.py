import importlib
import sys
import traceback

import click
import torch

from .api import audit as run_audit
from .config import Severity
from .core import Phase
from .reporters.console import ConsoleReporter
from .reporters.json import JSONReporter
from .reporters.sarif import SARIFReporter


def load_model_from_string(import_str: str) -> torch.nn.Module:
    """
    Dynamically imports a model from a string like 'my_project.models:ResNet'.
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


# Helper to parse rule lists
def parse_rule_list(ctx, param, value):
    if not value:
        return set()
    rules = set()
    for item in value:
        rules.update(item.split(","))
    return rules


@click.command()
@click.argument("target", required=True)
@click.option(
    "--format", "-f",
    type=click.Choice(["rich", "json", "sarif"], case_sensitive=False),
    multiple=True,
    default=["rich"],
    help="Output format(s). Can be specified multiple times. Default: rich."
)
@click.option(
    "--output", "-o",
    type=click.Path(writable=True),
    help="Output file for machine-readable formats (JSON/SARIF)."
)
@click.option(
    "--fail-level",
    type=click.Choice(["INFO", "WARN", "ERROR", "FATAL"], case_sensitive=False),
    default="ERROR",
    help="Exit with non-zero status if findings meet this severity.",
)
@click.option(
    "--phase",
    type=click.Choice([p.value for p in Phase], case_sensitive=False),
    default="static",
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
    callback=parse_rule_list,
    help="Only run specific rules (by ID). Can be comma-separated."
)
@click.option(
    "--ignore",
    multiple=True,
    callback=parse_rule_list,
    help="Ignore specific rules (by ID). Can be comma-separated."
)
@click.option(
    "--show-suppressed",
    is_flag=True,
    help="Include suppressed findings in the output."
)
@click.option(
    "--ignore-internal-errors",
    is_flag=True,
    help="Suppress internal validator crashes (TA000)."
)
def main(
        target: str,
        format,
        output,
        fail_level: str,
        phase: str,
        step: int,
        baseline: str,
        update_baseline: bool,
        select,
        ignore,
        show_suppressed,
        ignore_internal_errors,
):
    """
    Audit a PyTorch model for stability and performance issues.
    TARGET should be an import string (e.g., 'torchvision.models:resnet18').
    """
    # 1. Load User Model
    click.secho(f"🔎 Loading target: {target}...", dim=True)
    try:
        model = load_model_from_string(target)
    except Exception as e:
        click.secho(f"FATAL: {e}", fg="red")
        sys.exit(1)

    # 2. Run Audit (via Library API)
    try:
        result = run_audit(
            model=model,
            step=step,
            phase=phase,
            fail_level=fail_level,
            show_report=False,
            baseline_file=baseline,
            update_baseline=update_baseline,
        )


    except Exception as e:
        click.secho(f"Error running audit: {e}", err=True)
        traceback.print_exc()
        sys.exit(1)

    # 3. Report Results
    formats = set(f.lower() for f in format)

    file_formats = {"json", "sarif"}
    requested_file_formats = formats.intersection(file_formats)
    if output and len(requested_file_formats) > 1:
        click.echo("Error: --output cannot be used with multiple file formats (JSON + SARIF).", err=True)
        sys.exit(1)

    if "rich" in formats:
        ConsoleReporter().report(result)

    if "json" in formats:
        JSONReporter(dest=output).report(result)

    if "sarif" in formats:
        SARIFReporter(dest=output).report(result)

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()