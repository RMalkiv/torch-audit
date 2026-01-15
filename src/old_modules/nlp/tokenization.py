from typing import Any, List, Optional

import torch
import torch.nn as nn

from ...core_old.issue import AuditIssue
from ...core_old.validator import Validator


class TokenizationValidator(Validator):
    """
    Monitors the token stream entering Embedding layers.
    - Detects wasted compute (High Padding Ratio).
    - Detects tokenization quality issues (High UNK rate).
    - Detects crash risks (Token IDs > Vocab Size).
    """

    def __init__(
        self,
        pad_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ):
        self.pad_id = pad_token_id
        self.unk_id = unk_token_id
        self.vocab_size = vocab_size
        self.hooks = []
        self.reset_stats()

    def reset_stats(self):
        self.total_tokens = 0
        self.pad_counts = 0
        self.unk_counts = 0
        self.out_of_bounds_counts = 0

    def attach(self, model: nn.Module):
        self.reset_stats()
        self.hooks = []
        for _, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _hook_fn(self, module, input, output):
        x = input[0]
        if not isinstance(x, torch.Tensor) or x.dim() < 2:
            return

        num_tokens = x.numel()
        self.total_tokens += num_tokens
        if self.pad_id is not None:
            self.pad_counts += (x == self.pad_id).sum().item()
        if self.unk_id is not None:
            self.unk_counts += (x == self.unk_id).sum().item()
        if self.vocab_size is not None:
            self.out_of_bounds_counts += (x >= self.vocab_size).sum().item()

    def check_data(self, batch: Any) -> List[AuditIssue]:
        """
        Verifies that attention_mask correctly masks padding tokens.
        """
        issues = []
        if self.pad_id is None:
            return issues

        input_ids = None
        attention_mask = None

        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
        elif hasattr(batch, "input_ids") and hasattr(batch, "attention_mask"):
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask

        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            return issues

        is_pad = input_ids == self.pad_id
        is_masked = attention_mask == 0

        mismatches = is_pad != is_masked
        if mismatches.any():
            mismatch_count = mismatches.sum().item()
            total = input_ids.numel()
            ratio = mismatch_count / total

            issues.append(
                AuditIssue(
                    type="Data Integrity",
                    layer="Input Batch",
                    message=f"Attention Mask mismatch detected on {mismatch_count} tokens ({ratio:.1%}). "
                    f"The `attention_mask` zeros do not align with `input_ids` padding ({self.pad_id}). "
                    f"The model may be attending to padding or ignoring real words.",
                    severity="ERROR",
                )
            )

        return issues

    def check_dynamic(self, model: nn.Module) -> List[AuditIssue]:
        issues = []
        if self.total_tokens == 0:
            return issues

        total = float(self.total_tokens)

        # 1. Check Padding Efficiency
        if self.pad_id is not None:
            pad_ratio = float(self.pad_counts) / total
            if pad_ratio > 0.50:
                issues.append(
                    AuditIssue(
                        type="NLP Efficiency",
                        layer="Embedding",
                        message=f"High Padding detected ({pad_ratio:.1%} of tokens). "
                        f"More than 50% of your compute is wasted processing padding. "
                        f"Consider using dynamic padding or bucketing.",
                        severity="WARNING",
                    )
                )

        # 2. Check Tokenizer Quality (UNKs)
        if self.unk_id is not None:
            unk_ratio = float(self.unk_counts) / total
            if unk_ratio > 0.05:
                issues.append(
                    AuditIssue(
                        type="NLP Tokenization",
                        layer="Embedding",
                        message=f"High [UNK] rate ({unk_ratio:.1%}). "
                        f"Your tokenizer might be mismatched with the data.",
                        severity="ERROR",
                    )
                )

        # 3. Check Crash Risk (OOB)
        if self.vocab_size is not None and self.out_of_bounds_counts > 0:
            issues.append(
                AuditIssue(
                    type="NLP Crash Risk",
                    layer="Embedding",
                    message=f"Found {int(self.out_of_bounds_counts)} tokens >= vocab_size ({self.vocab_size}). "
                    f"This will cause an 'Embedding index out of range' crash.",
                    severity="ERROR",
                )
            )

        return issues
