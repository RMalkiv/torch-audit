"""Torch-Audit demo: computer vision-ish model + common input mistakes.

Run:
  python examples/demo_cv.py

This demo focuses on issues that the built-in validators *already* detect:
  - Conv architecture hints (even kernel sizes, redundant bias before BatchNorm,
    dead filters)
  - Data checks (NHWC vs NCHW, suspicious float ranges, tiny batch with BatchNorm)
  - Runtime graph checks (unused/zombie layers)
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


class BadConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d(kernel_size=2) => TA404 (even kernel)
        # bias=True followed by BatchNorm => TA400 (redundant bias)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(16 * 31 * 31, 10)

        # Unused layer to trigger TA500 after a forward pass.
        self.ghost = nn.Conv2d(16, 16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def run_demo() -> None:
    print("\n" + "=" * 72)
    print("🖼️  TORCH-AUDIT: CV DEMO (current API)")
    print("=" * 72)

    torch.manual_seed(0)

    model = BadConvNet()

    # Make some filters exactly zero to trigger TA405.
    with torch.no_grad():
        conv = model.features[0]
        conv.weight.data[3:] = 0.0

    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    auditor = Auditor(
        model, optimizer=optimizer, every_n_steps=1, reporters=[ConsoleReporter()]
    )

    print("\n[1] Static / init audits")
    with auditor:
        auditor.audit_static()
        auditor.audit_init()

    print("\n[2] Data-only checks for common CV batch mistakes")
    # NHWC shaped tensor interpreted as NCHW: (N, 32, 32, 3)
    # Heuristic triggers when 'C' is large and 'W' is tiny.
    bad_layout = torch.randn(2, 32, 32, 3)

    # Float range like [0..255] -> TA301
    bad_range = torch.rand(2, 3, 32, 32) * 255.0

    with auditor:
        auditor.audit_data({"nhwc": bad_layout, "range": bad_range})

    print("\n[3] A tiny training step (runtime audits)")
    # Proper NCHW input, but still unnormalized to keep TA301 visible.
    x = torch.rand(2, 3, 32, 32) * 255.0
    y = torch.randint(0, 10, (2,))
    criterion = nn.CrossEntropyLoss()

    with auditor:
        optimizer.zero_grad(set_to_none=True)
        logits = auditor.forward(x)
        loss = criterion(logits, y)
        auditor.backward(loss)
        auditor.optimizer_step()

    # Final report.
    auditor.finish(report=True)


if __name__ == "__main__":
    run_demo()
