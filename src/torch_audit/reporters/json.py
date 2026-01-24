import dataclasses
import json
from typing import Any, TextIO

from torch_audit import __version__

from ..core import AuditResult, Phase, Severity
from .base import Reporter


class JSONReporter(Reporter):
    def __init__(self, dest: str | TextIO | None = None):
        """
        Args:
            dest: File path (str) or open stream (TextIO) to write JSON to.
                  Defaults to stdout (None).
        """
        self.dest = dest

    def report(self, result: AuditResult) -> None:
        class AuditEncoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, (Severity, Phase)):
                    return o.value
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                return super().default(o)

        # 1. Deterministic Sort
        sorted_findings = sorted(
            result.findings,
            key=lambda f: (f.rule_id, f.module_path or "", f.entity or "", f.message),
        )

        # 2. Extract Active Rules Metadata
        active_rule_ids = {f.rule_id for f in sorted_findings}
        rules_meta = {}
        for rid in sorted(active_rule_ids):
            if rid in result.rules:
                rules_meta[rid] = result.rules[rid].to_dict()

        output = {
            "meta": {"tool": "torch-audit", "version": __version__},
            "summary": {
                "exit_code": result.exit_code,
                "max_severity": result.max_severity.value,
                "max_severity_all": result.max_severity_all.value,
                "new_findings_count": result.new_findings_count,
                "baseline_applied": result.baseline_applied,
                "total_findings": len(result.findings),
                "suppressed_findings": result.suppressed_count,
            },
            "rules": rules_meta,
            "findings": sorted_findings,
        }

        # 3. Write Output
        if self.dest is None:
            print(json.dumps(output, cls=AuditEncoder, indent=2))
        elif isinstance(self.dest, str):
            with open(self.dest, "w") as f:
                json.dump(output, f, cls=AuditEncoder, indent=2)
        else:
            # Assume it's a stream (TextIO)
            json.dump(output, self.dest, cls=AuditEncoder, indent=2)
