"""Torch-Audit demo: simple NLP-ish model.

Run:
  python examples/demo_nlp.py

Highlights:
  - Data hygiene checks for integer token tensors (e.g., negative indices)
  - Optimizer checks for weight decay on embeddings
  - Runtime graph checks (unused layers) and stability checks around backward
"""

import sys
from pathlib import Path

# Allow running this demo without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
import torch.optim as optim

from torch_audit import Auditor
from torch_audit.reporters.console import ConsoleReporter


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_model, vocab_size)

        # Unused layer (TA500) once a forward pass happens.
        self.ghost = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        x = self.embed(input_ids).mean(dim=1)
        x = self.proj(x)
        return self.head(x)


def run_demo() -> None:
    print("\n" + "=" * 72)
    print("📖 TORCH-AUDIT: NLP DEMO (current API)")
    print("=" * 72)

    torch.manual_seed(0)

    model = TinyLM(vocab_size=128, d_model=64)

    # Intentionally use Adam + weight_decay to trigger:
    #   - TA401 (Adam vs AdamW)
    #   - TA403 (weight decay on embeddings)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)

    auditor = Auditor(
        model, optimizer=optimizer, every_n_steps=1, reporters=[ConsoleReporter()]
    )

    # Batch with a negative token ID triggers TA305 in DataValidator.
    bad_input_ids = torch.tensor(
        [
            [5, 10, -1, 7, 0],
            [3, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    print("\n[1] Data-only audit catches invalid token IDs before a crash")
    with auditor:
        auditor.audit_data({"input_ids": bad_input_ids})

    print("\n[2] Tiny training step with a valid batch (runtime audits)")
    input_ids = torch.randint(0, 128, (4, 8), dtype=torch.long)
    targets = torch.randint(0, 128, (4,), dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    with auditor:
        auditor.audit_static()
        auditor.audit_init()

        optimizer.zero_grad(set_to_none=True)
        logits = auditor.forward(input_ids)
        loss = criterion(logits, targets)
        auditor.backward(loss)
        auditor.optimizer_step()

    auditor.finish(report=True)


if __name__ == "__main__":
    run_demo()
