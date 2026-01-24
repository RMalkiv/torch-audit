from typing import Protocol, runtime_checkable

from ..core import AuditResult


@runtime_checkable
class Reporter(Protocol):
    def report(self, result: AuditResult) -> None:
        """
        Render the audit result to the destination (stdout, file, API).
        """
        pass
