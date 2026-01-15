import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Set

from .core import Finding, Severity


@dataclass
class Suppression:
    rule_id: str
    reason: str
    module_regex: Optional[str] = None

    _regex: Optional[Pattern] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._regex = None
        if self.module_regex:
            try:
                self._regex = re.compile(self.module_regex)
            except re.error as e:
                raise ValueError(
                    f"Invalid suppression regex for Rule '{self.rule_id}': '{self.module_regex}'. "
                    f"Regex error: {e}"
                ) from e

    def matches(self, finding: Finding) -> bool:
        if finding.rule_id != self.rule_id:
            return False

        if self._regex:
            if not finding.module_path:
                return False
            if not self._regex.search(finding.module_path):
                return False

        return True


@dataclass
class AuditConfig:
    fail_level: Severity = Severity.ERROR

    suppress_internal_errors: bool = False
    suppressions: List[Suppression] = field(default_factory=list)

    baseline_file: Optional[str] = None
    update_baseline: bool = False

    # New CLI Options
    select_rules: Optional[Set[str]] = None
    ignore_rules: Optional[Set[str]] = None
    show_suppressed: bool = False

    def is_failure(self, severity: Severity) -> bool:
        return severity >= self.fail_level

    def is_rule_allowed(self, rule_id: str) -> bool:
        """Checks --select and --ignore logic."""
        if self.select_rules and rule_id not in self.select_rules:
            return False
        if self.ignore_rules and rule_id in self.ignore_rules:
            return False
        return True

    def is_suppressed(self, finding: Finding) -> bool:
        """Checks if a finding matches any suppression rule."""
        for suppression in self.suppressions:
            if suppression.matches(finding):
                return True
        return False