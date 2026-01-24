import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .utils.fingerprints import stable_fingerprint


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"

    def _rank(self) -> int:
        return {"INFO": 0, "WARN": 1, "ERROR": 2, "FATAL": 3}[self.value]

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self._rank() < other._rank()
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self._rank() <= other._rank()
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self._rank() > other._rank()
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self._rank() >= other._rank()
        return NotImplemented


class Phase(str, Enum):
    STATIC = "static"
    INIT = "init"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    description: str
    remediation: str
    category: str
    default_severity: Severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "remediation": self.remediation,
            "category": self.category,
            "default_severity": self.default_severity.value,
        }


@dataclass
class Finding:
    rule_id: str
    message: str
    severity: Severity

    module_path: str | None = None
    entity: str | None = None
    step: int | None = None
    phase: Phase = Phase.STATIC

    metadata: dict[str, Any] = field(default_factory=dict)

    # New field to support --show-suppressed
    suppressed: bool = False

    def get_fingerprint(self) -> str:
        """Stable identity: v{VER}:{rule}::{module}::{entity}"""
        return stable_fingerprint(self.rule_id, self.module_path, self.entity)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the finding to a dictionary.
        Handles Enum conversion (Severity/Phase) automatically.
        """

        def _factory(data):
            return {k: (v.value if isinstance(v, Enum) else v) for k, v in data}

        return dataclasses.asdict(self, dict_factory=_factory)


@dataclass
class AuditResult:
    findings: list[Finding]
    exit_code: int
    max_severity: Severity
    max_severity_all: Severity = Severity.INFO
    new_findings_count: int = 0
    baseline_applied: bool = False
    suppressed_count: int = 0
    rules: dict[str, Rule] = field(default_factory=dict)
