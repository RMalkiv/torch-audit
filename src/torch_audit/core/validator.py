from abc import ABC
from typing import List, Dict, Any
import torch.nn as nn

from torch_audit.core.issue import AuditIssue


class Validator(ABC):
    """
    Abstract Base Class for all audit validators.
    Subclasses can implement any combination of the check methods.
    """

    def attach(self, model: nn.Module):
        """
        Optional: Attach hooks to the model.
        Called by Auditor.start_dynamic_audit().
        """
        pass

    def detach(self):
        """
        Optional: Remove hooks.
        Called by Auditor.stop_dynamic_audit().
        """
        pass

    def check_static(self, model) -> List[AuditIssue]:
        """
        Run one-time analysis on model architecture or weights.
        (e.g., Check layer dimensions, weight initialization)
        """
        return []

    def check_data(self, batch: Any) -> List[Dict[str, Any]]:
        """
        Run checks on the input data batch.
        (e.g., Check for NaNs, normalization, shape)
        """
        return []

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        """
        Run checks at the end of a training step.
        This is where you inspect collected hook stats or gradients.
        """
        return []