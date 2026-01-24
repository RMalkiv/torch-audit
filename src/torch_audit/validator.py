from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Any

from .context import AuditContext
from .core import Finding, Phase, Rule


class BaseValidator(ABC):
    @property
    @abstractmethod
    def rule(self) -> Rule:
        """
        The primary rule associated with this validator.
        Used as the default if 'rules' is not overridden.
        """
        pass

    @property
    def rules(self) -> Sequence[Rule]:
        """
        Returns the list of rules this validator can emit.
        Defaults to [self.rule] for single-rule validators.
        """
        return [self.rule]

    @property
    def emits_rule_ids(self) -> set[str]:
        """Returns the set of Rule IDs this validator is allowed to emit."""
        return {r.id for r in self.rules}

    @property
    def supported_phases(self) -> set[Phase] | None:
        """
        Returns the set of phases this validator supports.
        If None, the validator is run in all phases.

        Override this to optimize performance by skipping irrelevant phases.
        """
        return None

    @abstractmethod
    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        pass

    # --- Optional lifecycle hooks (runtime auditing) ---
    #
    # Stateless validators can ignore these.
    # Stateful validators (e.g. forward-hook based) can override attach/detach
    # and/or on_phase_start/on_phase_end.
    def attach(self, model: Any) -> None:
        """Optional: attach instrumentation to a model (e.g., register hooks)."""

    def detach(self) -> None:
        """Optional: detach instrumentation from a model (e.g., remove hooks)."""

    def on_phase_start(self, context: AuditContext) -> None:
        """Optional: called before a phase runs (e.g., before forward/backward)."""

    def on_phase_end(self, context: AuditContext) -> None:
        """Optional: called after a phase runs (e.g., after forward/backward)."""
