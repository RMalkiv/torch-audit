import hashlib
import json
from typing import TextIO

from torch_audit import __version__

from ..core import AuditResult, Severity
from .base import Reporter


class SARIFReporter(Reporter):
    def __init__(
        self,
        dest: str | TextIO | None = None,
        location_file: str = "model.py",
        location_line: int = 1,
    ):
        """
        Args:
            dest: Output destination.
            location_file: Fallback filename for findings (default: model.py).
            location_line: Fallback line number (default: 1).
        """
        self.dest = dest
        self.location_file = location_file
        self.location_line = location_line

    def report(self, result: AuditResult) -> None:
        # 1. Deterministic Sort (keeps SARIF diffs stable in CI)
        sorted_findings = sorted(
            result.findings,
            key=lambda f: (f.rule_id, f.module_path or "", f.entity or "", f.message),
        )

        # 2. Build Rule Metadata & Index Map
        active_rule_ids = sorted({f.rule_id for f in sorted_findings})
        rules_meta = []
        rule_indices = {}

        for idx, rule_id in enumerate(active_rule_ids):
            rule_indices[rule_id] = idx
            rule = result.rules.get(rule_id)

            sarif_level = "warning"
            if rule and rule.default_severity in [Severity.ERROR, Severity.FATAL]:
                sarif_level = "error"
            elif rule and rule.default_severity == Severity.INFO:
                sarif_level = "note"

            meta = {
                "id": rule_id,
            }
            if rule:
                meta.update(
                    {
                        "name": rule.title,
                        "shortDescription": {"text": rule.title},
                        "fullDescription": {"text": rule.description},
                        "help": {
                            "text": f"{rule.description}\n\nRemediation: {rule.remediation}",
                            "markdown": f"**{rule.title}**\n\n{rule.description}\n\n**Remediation**\n{rule.remediation}",
                        },
                        "defaultConfiguration": {"level": sarif_level},
                        "properties": {"category": rule.category},
                    }
                )
            rules_meta.append(meta)

        # 2. Map Findings
        results = []
        for f in sorted_findings:
            level = "warning"
            if f.severity in [Severity.ERROR, Severity.FATAL]:
                level = "error"
            elif f.severity == Severity.INFO:
                level = "note"

            fingerprint = f.get_fingerprint()

            hash_payload = f"{self.location_file}|{fingerprint}".encode()
            primary_hash = hashlib.sha256(hash_payload).hexdigest()

            results.append(
                {
                    "ruleId": f.rule_id,
                    "ruleIndex": rule_indices.get(f.rule_id, -1),
                    "level": level,
                    "message": {"text": f.message},
                    "fingerprints": {"stableFingerprint": fingerprint},
                    "partialFingerprints": {
                        "primaryLocationLineHash": primary_hash,
                        "torchAuditFingerprint": fingerprint,
                    },
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": self.location_file},
                                "region": {"startLine": self.location_line},
                            },
                            "logicalLocations": [
                                {
                                    "name": f.module_path or "global",
                                    "kind": (
                                        "function"
                                        if "grad" in (f.entity or "")
                                        else "variable"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

        sarif_log = {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "torch-audit",
                            "semanticVersion": __version__,
                            "rules": rules_meta,
                        }
                    },
                    "results": results,
                }
            ],
        }

        # 3. Write Output
        output_str = json.dumps(sarif_log, indent=2)
        if self.dest is None:
            print(output_str)
        elif isinstance(self.dest, str):
            with open(self.dest, "w") as f:
                f.write(output_str)
        else:
            self.dest.write(output_str)
