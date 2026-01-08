import torch
import torch.nn as nn
from torch_audit import Auditor, AuditConfig


class BadNLPModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        # Issue 1: Config has pad_id=0, but Embedding has padding_idx=None
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        # Issue 2: Head not tied to embeddings
        self.head = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.head(x[:, -1, :])


def run_demo():
    print("\n" + "=" * 60)
    print("📖 TORCH-AUDIT: NLP DEMO")
    print("=" * 60)

    vocab_size = 1000
    pad_id = 0

    model = BadNLPModel(vocab_size)

    config = AuditConfig(
        monitor_nlp=True,
        pad_token_id=pad_id,
        vocab_size=vocab_size
    )

    auditor = Auditor(model, config=config)

    print("\n🔍 Static Audit...")
    auditor.audit_static()

    print("\n[Simulating Bad Batch]...")

    # Issue 3: Attention Mask Mismatch
    # We have padding (0) in input_ids, but mask is all 1s (Attention!)
    batch = {
        'input_ids': torch.tensor([
            [5, 10, 22, 0, 0],  # Padding at end
            [5, 0, 0, 0, 0]
        ]),
        'attention_mask': torch.tensor([
            [1, 1, 1, 1, 1],  # Should be [1, 1, 1, 0, 0]
            [1, 1, 1, 1, 1]  # Should be [1, 0, 0, 0, 0]
        ])
    }

    with auditor.audit_dynamic():
        # Captures mask mismatch
        auditor.audit_data(batch)

        model(batch['input_ids'])


if __name__ == "__main__":
    run_demo()
