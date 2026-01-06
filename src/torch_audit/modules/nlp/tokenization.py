import torch
import torch.nn as nn
from typing import Dict, List, Optional


class TokenizationMonitor:
    def __init__(
            self,
            pad_token_id: Optional[int] = None,
            unk_token_id: Optional[int] = None,
            vocab_size: Optional[int] = None
    ):
        self.pad_id = pad_token_id
        self.unk_id = unk_token_id
        self.vocab_size = vocab_size
        self.hooks = []

        self.total_tokens = 0
        self.pad_counts = 0
        self.unk_counts = 0
        self.out_of_bounds_counts = 0

    def _hook_fn(self, module, input, output):
        x = input[0]
        if not isinstance(x, torch.Tensor):
            return

        if x.dim() < 2:
            return

        num_tokens = x.numel()
        self.total_tokens += num_tokens

        if self.pad_id is not None:
            self.pad_counts += (x == self.pad_id).sum()

        if self.unk_id is not None:
            self.unk_counts += (x == self.unk_id).sum()

        if self.vocab_size is not None:
            self.out_of_bounds_counts += (x >= self.vocab_size).sum()

    def attach(self, model: nn.Module):
        self.total_tokens = 0
        self.pad_counts = 0
        self.unk_counts = 0
        self.out_of_bounds_counts = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_issues(self, model: nn.Module) -> List[Dict]:
        issues = []
        if self.total_tokens == 0:
            return issues

        total = float(self.total_tokens)

        if self.pad_id is not None:
            pad_ratio = float(self.pad_counts) / total
            if pad_ratio > 0.50:
                issues.append({
                    "type": "NLP Efficiency",
                    "layer": "Embedding",
                    "message": f"High Padding detected ({pad_ratio:.1%} of tokens). "
                               f"50%+ of your compute is wasted processing padding. "
                               f"Consider using dynamic padding or bucketing.",
                    "severity": "WARNING"
                })

        if self.unk_id is not None:
            unk_ratio = float(self.unk_counts) / total
            if unk_ratio > 0.05:
                issues.append({
                    "type": "NLP Tokenization",
                    "layer": "Embedding",
                    "message": f"High [UNK] rate ({unk_ratio:.1%}). "
                               f"Your tokenizer might be mismatched with the data language.",
                    "severity": "ERROR"
                })

        if self.vocab_size is not None and self.out_of_bounds_counts > 0:
            issues.append({
                "type": "NLP Crash Risk",
                "layer": "Embedding",
                "message": f"Found {int(self.out_of_bounds_counts)} tokens > vocab_size ({self.vocab_size}). "
                           f"This will cause an 'Embedding index out of range' crash.",
                "severity": "ERROR"
            })

        return issues
