from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["ERROR", "WARNING", "INFO"]


@dataclass(order=True)
class AuditIssue:
    sort_index: int = field(init=False, repr=False)

    type: str = field(compare=False)
    message: str = field(compare=False)
    layer: str = field(compare=False)
    severity: Severity = field(default="WARNING", compare=False)

    def __post_init__(self):
        priorities = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        self.sort_index = priorities.get(self.severity, 99)