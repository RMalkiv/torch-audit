"""Torch-Audit demo: plain PyTorch training loop.

This demo is intentionally self-contained and matches the *current* public API.

Run:
  python examples/demo_general.py

What it demonstrates:
  - Static + init audits (architecture, hardware, optimizer config)
  - Runtime audits during forward/backward/optimizer phases
  - `audit_step(...)` decorator for a function-style training step
  - `autopatch(...)` for zero-touch auditing (no wrappers)
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

from torch_audit import Auditor, audit_step, autopatch
from torch_audit.reporters.console import ConsoleReporter


class BrokenMLP(nn.Module):
    """A tiny model with intentionally bad patterns to trigger audit findings."""

    def __init__(self):
        super().__init__()

        # 1) Tensor Core misalignment (64 -> 127) and redundant bias before BatchNorm.
        #    (TA200, TA400)
        self.features = nn.Sequential(
            nn.Linear(64, 127, bias=True),
            nn.BatchNorm1d(127),
            nn.ReLU(),
        )

        self.head = nn.Linear(127, 10)

        # 2) Zombie/unused layer (never called in forward).
        #    (TA500)
        self.ghost = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def run_demo() -> None:
    print("\n" + "=" * 72)
    print("🔥 TORCH-AUDIT: GENERAL DEMO (current API)")
    print("=" * 72)

    torch.manual_seed(0)

    model = BrokenMLP()

    # 3) Bad init to provoke activation collapse (dead ReLU) and zero grads.
    #    We'll also keep inputs strictly positive so the first Linear outputs negative.
    with torch.no_grad():
        lin = model.features[0]
        lin.weight.fill_(-1.0)
        lin.bias.fill_(-1.0)

    # 4) Optimizer config issues: Adam + weight_decay (TA401) and weight decay on
    #    bias / BatchNorm params (TA402).
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Attach a reporter; we'll call finish(report=True) at the end.
    auditor = Auditor(
        model,
        optimizer=optimizer,
        every_n_steps=1,
        reporters=[ConsoleReporter()],
    )

    criterion = nn.CrossEntropyLoss()

    # Inputs intentionally look like unnormalized [0..255] floats to trigger TA301.
    batch_size = 4  # < 8 + BatchNorm => TA304
    x = torch.rand(batch_size, 64) * 255.0
    y = torch.randint(0, 10, (batch_size,))

    print("\n[Scenario A] Explicit forward/backward/optimizer wrappers")
    with auditor:  # attaches hook-based validators
        # One-shot audits
        auditor.audit_static()
        auditor.audit_init()

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = auditor.forward(x)
            loss = criterion(logits, y)
            auditor.backward(loss)
            auditor.optimizer_step()

            # Optional: stream findings during the loop
            new_findings = auditor.pop_new_findings()
            if new_findings:
                print(f"  step={step}: +{len(new_findings)} findings")

    # Emit a final report to console.
    auditor.finish(report=True)

    print("\n[Scenario B] Decorator for a function-style train step")

    # A fresh auditor for the decorator example (keeps output readable).
    model2 = BrokenMLP()
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-2)
    auditor2 = Auditor(model2, optimizer=optimizer2, every_n_steps=1)

    @audit_step(auditor2)
    def train_step(batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        optimizer2.zero_grad(set_to_none=True)
        logits = model2(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer2.step()
        return loss

    with auditor2:
        auditor2.audit_static()
        auditor2.audit_init()
        _ = train_step(x, y)

    # Print report for the decorator scenario.
    ConsoleReporter().report(auditor2.finish())

    print("\n[Scenario C] Zero-touch mode via autopatch() (no wrappers)")

    model3 = BrokenMLP()
    optimizer3 = optim.AdamW(model3.parameters(), lr=3e-4)

    # This patches model3.forward and optimizer3.step in-place.
    auditor3 = autopatch(
        model3,
        optimizer=optimizer3,
        every_n_steps=1,
        reporters=[ConsoleReporter()],
        run_static=True,
        run_init=True,
    )

    # Normal training loop code (no auditor wrappers)
    optimizer3.zero_grad(set_to_none=True)
    logits3 = model3(x)
    loss3 = criterion(logits3, y)
    loss3.backward()
    optimizer3.step()

    auditor3.finish(report=True)
    auditor3.unpatch()


if __name__ == "__main__":
    run_demo()
