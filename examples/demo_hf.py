"""Torch-Audit demo: Hugging Face Transformers (no downloads required).

Run:
  python examples/demo_hf.py

Notes:
  - This demo only requires `transformers` if you want to run the HF section.
    If `transformers` is not installed, the script prints an instruction and exits.
  - We intentionally instantiate a tiny model from a config (no `from_pretrained`).
    That keeps it fast and avoids network access.

What it shows:
  - Auditing a Transformers model in a normal PyTorch training loop
  - Data checks on typical HF batches (input_ids / attention_mask / labels)
"""

import sys
from pathlib import Path

# Allow running this demo without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import torch
import torch.optim as optim

from torch_audit import Auditor
from torch_audit.reporters.console import ConsoleReporter


def run_demo() -> None:
    try:
        from transformers import BertConfig, BertForSequenceClassification
    except Exception:
        print("\n" + "=" * 72)
        print("🤗 TORCH-AUDIT: HUGGING FACE DEMO (current API)")
        print("=" * 72)
        print(
            "\nThis demo requires the optional dependency `transformers`.\n"
            "Install it and re-run:\n"
            "  pip install transformers\n"
        )
        return

    print("\n" + "=" * 72)
    print("🤗 TORCH-AUDIT: HUGGING FACE DEMO (current API)")
    print("=" * 72)

    torch.manual_seed(0)

    # Tiny BERT config (fast, no downloads).
    config = BertConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)
    auditor = Auditor(
        model, optimizer=optimizer, every_n_steps=1, reporters=[ConsoleReporter()]
    )

    # Create a batch. We'll include one intentionally bad input_ids tensor to show
    # TA305 (negative integer inputs) is caught before embedding lookup.
    bad_batch = {
        "input_ids": torch.tensor([[5, 10, -1, 7, 0, 0, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long),
        "labels": torch.tensor([1], dtype=torch.long),
    }

    print("\n[1] Data-only audit on an invalid batch (no forward pass)")
    with auditor:
        auditor.audit_data(bad_batch)

    # Now a valid batch for a tiny training step.
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (4, 16), dtype=torch.long),
        "attention_mask": torch.ones((4, 16), dtype=torch.long),
        "labels": torch.randint(0, 2, (4,), dtype=torch.long),
    }

    print("\n[2] Tiny training step with Auditor wrappers")
    with auditor:
        auditor.audit_static()
        auditor.audit_init()

        optimizer.zero_grad(set_to_none=True)
        out = auditor.forward(**batch)
        loss = out.loss
        auditor.backward(loss)
        auditor.optimizer_step()

    auditor.finish(report=True)


if __name__ == "__main__":
    run_demo()
