import torch
import torch.nn as nn
from torch_audit import Auditor


class BadNLPModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super().__init__()
        # Issue: Padding Index is NOT set
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)

        # Issue: Output head not tied to embeddings
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
    unk_id = 100

    model = BadNLPModel(vocab_size)

    config = {
        'monitor_nlp': True,
        'pad_token_id': pad_id,
        'unk_token_id': unk_id,
        'vocab_size': vocab_size
    }

    auditor = Auditor(model, config=config)

    print("\n🔍 Starting Static Audit...")
    auditor.audit_static()

    print("\n[Simulating Bad Batch]...")

    # Row 1: Normal
    # Row 2 & 3: 90% Padding
    input_ids = torch.tensor([
        [5, 10, 22, 33, 44, 55, 66, 77, 88, 99],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Inject UNK tokens to trigger that check too
    input_ids[0, 5:] = 100

    with auditor.audit_dynamic():
        output = model(input_ids)

        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    run_demo()