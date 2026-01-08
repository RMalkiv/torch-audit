import torch.nn as nn
from typing import List, Optional
from ...core.validator import Validator
from ...core.issue import AuditIssue


class StructureValidator(Validator):
    """
    Static checks for NLP Model Architecture.
    - Checks Weight Tying across multiple embeddings (Seq2Seq, Multi-domain).
    - Checks for configuration mismatches.
    """

    def __init__(self, vocab_size: Optional[int] = None, pad_token_id: Optional[int] = None):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def check_static(self, model: nn.Module) -> List[AuditIssue]:
        issues = []

        # 1. Check Padding Configuration
        self._check_padding_config(issues, model)

        # 2. Check Weight Tying (Robust Multi-Vocab Logic)
        self._check_weight_tying(issues, model)

        return issues

    def _check_padding_config(self, issues: List[AuditIssue], model: nn.Module):
        if self.pad_token_id is None:
            return

        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                if module.num_embeddings < 100:
                    continue

                if module.padding_idx is None:
                    issues.append(AuditIssue(
                        type="NLP Efficiency",
                        layer=name,
                        message=f"Config defines `pad_token_id={self.pad_token_id}`, but this Embedding has `padding_idx=None`. "
                                f"Gradients for padding will be calculated unnecessarily.",
                        severity="WARNING"
                    ))
                elif module.padding_idx != self.pad_token_id:
                    issues.append(AuditIssue(
                        type="Configuration Mismatch",
                        layer=name,
                        message=f"Config defines `pad_token_id={self.pad_token_id}`, "
                                f"but Embedding has `padding_idx={module.padding_idx}`.",
                        severity="ERROR"
                    ))

    def _check_weight_tying(self, issues: List[AuditIssue], model: nn.Module):
        """
        Finds pairs of (Embedding, Linear) that share dimensions and checks if weights are tied.
        """
        embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding) and m.num_embeddings > 1000]
        linear_heads = [m for m in model.modules() if isinstance(m, nn.Linear)]

        for embed in embeddings:
            v_size = embed.num_embeddings
            d_model = embed.embedding_dim

            compatible_heads = [
                h for h in linear_heads
                if h.out_features == v_size and h.in_features == d_model
            ]

            if not compatible_heads:
                continue

            is_tied = False
            for head in compatible_heads:
                if head.weight is embed.weight:
                    is_tied = True
                    break

            if not is_tied:
                issues.append(AuditIssue(
                    type="NLP Optimization",
                    layer="Architecture",
                    message=f"Found Embedding ({d_model}->{v_size}) and compatible Output Head ({d_model}->{v_size}) "
                            f"that are NOT tied. "
                            f"If this is a language model head, consider tying weights for better convergence: "
                            f"`head.weight = embedding.weight`.",
                    severity="INFO"
                ))
