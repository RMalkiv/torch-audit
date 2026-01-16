from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-audit")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]

from .api import audit
from .core import Phase, Severity, Finding, Rule, AuditResult
from .runtime import Auditor, audit_dynamic, audit_step

__all__ += [
    "audit",
    "Auditor",
    "audit_dynamic",
    "audit_step",
    "Phase",
    "Severity",
    "Finding",
    "Rule",
    "AuditResult",
]
