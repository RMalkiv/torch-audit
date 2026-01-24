import argparse
from pathlib import Path


def _load_rules() -> list[dict[str, str]]:
    """Load built-in rules from the package registry.

    We add the repository root's `src/` to sys.path so this script works
    without installing the package.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    # Importing validators registers rules in the global registry.
    from torch_audit.loader import load_runtime_validators
    from torch_audit.registry import RuleRegistry

    load_runtime_validators()
    rules = RuleRegistry.all_rules()

    out: list[dict[str, str]] = []
    for r in rules:
        out.append(
            {
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "severity": r.default_severity.value,
                "description": r.description,
                "remediation": r.remediation,
            }
        )
    return out


def _render_markdown(rows: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Rules")
    lines.append("")
    lines.append(
        "This is an auto-generated list of built-in **torch-audit** rules. "
        "To regenerate: `python scripts/generate_rules.py`"
    )
    lines.append("")
    lines.append("## Rule packs")
    lines.append("")
    lines.append("- `TA1xx` — Stability")
    lines.append("- `TA2xx` — Hardware")
    lines.append("- `TA3xx` — Data integrity")
    lines.append("- `TA4xx` — Architecture & optimization")
    lines.append("- `TA5xx` — Runtime graph / hooks")
    lines.append("")
    lines.append("## Index")
    lines.append("")
    lines.append("| ID | Default severity | Category | Title |")
    lines.append("|---:|:-----------------|:---------|:------|")
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['severity']} | {r['category']} | {r['title']} |"
        )

    lines.append("")
    lines.append("## Details")
    lines.append("")

    for r in rows:
        lines.append(f"### {r['id']} — {r['title']}")
        lines.append("")
        lines.append(f"- **Category:** {r['category']}")
        lines.append(f"- **Default severity:** {r['severity']}")
        lines.append("")
        lines.append("**Description**")
        lines.append("")
        lines.append(r["description"])
        lines.append("")
        lines.append("**Remediation**")
        lines.append("")
        lines.append(r["remediation"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RULES.md from registry")
    parser.add_argument(
        "--output",
        "-o",
        default="RULES.md",
        help="Output path (default: RULES.md)",
    )
    args = parser.parse_args()

    rows = _load_rules()
    # Stable ordering
    rows = sorted(rows, key=lambda r: r["id"])

    md = _render_markdown(rows)
    Path(args.output).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
