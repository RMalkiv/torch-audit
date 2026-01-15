from abc import ABC, abstractmethod
from typing import Generator, Optional, Sequence, Set

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
    def emits_rule_ids(self) -> Set[str]:
        """Returns the set of Rule IDs this validator is allowed to emit."""
        return {r.id for r in self.rules}

    @property
    def supported_phases(self) -> Optional[Set[Phase]]:
        """
        Returns the set of phases this validator supports.
        If None, the validator is run in all phases.

        Override this to optimize performance by skipping irrelevant phases.
        """
        return None

    @abstractmethod
    def check(self, context: AuditContext) -> Generator[Finding, None, None]:
        pass