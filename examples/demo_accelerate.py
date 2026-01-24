"""Torch-Audit demo: Hugging Face Accelerate integration (optional dependency).

Run:
  python examples/demo_accelerate.py

If `accelerate` is not installed, the script falls back to a single-process
PyTorch loop.

What this demonstrates:
  - Using Torch-Audit with an `Accelerator`-prepared model/optimizer
  - A practical way to handle backward auditing when using `accelerator.backward(loss)`

Notes:
  - This is a *demo* pattern. For a real project you may want to:
      - gate reporting to rank 0
      - tune `every_n_steps`
      - optionally disable hook-based validators if you need lower overhead
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
from torch.utils.data import DataLoader, TensorDataset

from torch_audit import Auditor
from torch_audit.core import Phase
from torch_audit.reporters.console import ConsoleReporter


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Unused layer (TA500)
        self.ghost = nn.Linear(16, 16)

    def forward(self, x):
        return self.net(x)


def _fallback_single_process() -> None:
    print("\n" + "=" * 72)
    print("🚀 TORCH-AUDIT: ACCELERATE DEMO (fallback PyTorch loop)")
    print("=" * 72)
    print(
        "\n`accelerate` is not installed in this environment.\n"
        "Install it and re-run for the full demo:\n"
        "  pip install accelerate\n"
    )

    torch.manual_seed(0)
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    auditor = Auditor(
        model, optimizer=optimizer, every_n_steps=1, reporters=[ConsoleReporter()]
    )

    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    loader = DataLoader(dataset, batch_size=8)
    loss_fn = nn.MSELoss()

    with auditor:
        auditor.audit_static()
        auditor.audit_init()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = auditor.forward(x)
            loss = loss_fn(pred, y)
            auditor.backward(loss)
            auditor.optimizer_step()
            break

    auditor.finish(report=True)


def run_demo() -> None:
    try:
        from accelerate import Accelerator  # type: ignore
    except Exception:
        _fallback_single_process()
        return

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("\n" + "=" * 72)
        print("🚀 TORCH-AUDIT: ACCELERATE DEMO (current API)")
        print("=" * 72)

    torch.manual_seed(0)

    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    dataset = TensorDataset(torch.randn(64, 10), torch.randn(64, 1))
    dataloader = DataLoader(dataset, batch_size=8)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # IMPORTANT: when using Accelerate, the "right" backward call is:
    #   accelerator.backward(loss)
    # so we run the BACKWARD audit manually right after that.
    auditor = Auditor(model, optimizer=optimizer, every_n_steps=2)

    loss_fn = nn.MSELoss()
    model.train()

    with auditor:
        auditor.audit_static()
        auditor.audit_init()

        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            # We can still use auditor.forward to run forward-phase audits.
            pred = auditor.forward(x)
            loss = loss_fn(pred, y)

            accelerator.backward(loss)

            # Manual backward audit (because we didn't call auditor.backward).
            ctx = auditor._make_context(Phase.BACKWARD, step=auditor.step, batch=(x, y))
            if auditor._should_audit(ctx.step, ctx.phase):
                auditor.runner.run_step(ctx)

            optimizer.step()

            # Manual optimizer-phase audit (because we didn't call auditor.optimizer_step).
            ctx2 = auditor._make_context(
                Phase.OPTIMIZER, step=auditor.step, batch=(x, y)
            )
            if auditor._should_audit(ctx2.step, ctx2.phase):
                auditor.runner.run_step(ctx2)

            auditor.step += 1

            if step >= 3:
                break

    # Only print report on main process to avoid log spam.
    if accelerator.is_main_process:
        ConsoleReporter().report(auditor.finish())


if __name__ == "__main__":
    run_demo()
