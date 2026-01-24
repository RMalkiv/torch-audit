"""Torch-Audit demo: Lightning integration (optional dependency).

Run:
  python examples/demo_lightning.py

If Lightning is not installed, the script falls back to a plain PyTorch loop.

Why this file exists:
  The repository currently does *not* ship a built-in Lightning callback module.
  This demo shows a minimal pattern you can copy-paste into your own codebase.
"""

import sys
from pathlib import Path
from typing import Any

# Allow running this demo without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch_audit import Auditor
from torch_audit.core import Phase
from torch_audit.reporters.console import ConsoleReporter


def _import_lightning():
    """Return the Lightning module namespace or None if unavailable."""
    try:
        import lightning.pytorch as pl  # type: ignore

        return pl
    except Exception:
        try:
            import pytorch_lightning as pl  # type: ignore

            return pl
        except Exception:
            return None


class TinyLitNet(nn.Module):
    """A small net with an intentionally unused layer (TA500)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.ghost = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_fallback_pytorch() -> None:
    print("\n" + "=" * 72)
    print("⚡ TORCH-AUDIT: LIGHTNING DEMO (fallback PyTorch loop)")
    print("=" * 72)
    print(
        "\nLightning is not installed in this environment.\n"
        "Install one of the following and re-run for the full callback example:\n"
        "  pip install lightning\n"
        "  # or\n"
        "  pip install pytorch-lightning\n"
    )

    torch.manual_seed(0)

    model = TinyLitNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    auditor = Auditor(
        model, optimizer=optimizer, every_n_steps=1, reporters=[ConsoleReporter()]
    )

    # Tiny batch + BatchNorm -> TA304
    x = torch.randn(2, 32)
    y = torch.randn(2, 1)
    loss_fn = nn.MSELoss()

    with auditor:
        auditor.audit_static()
        auditor.audit_init()

        optimizer.zero_grad(set_to_none=True)
        pred = auditor.forward(x)
        loss = loss_fn(pred, y)
        auditor.backward(loss)
        auditor.optimizer_step()

    auditor.finish(report=True)


def run_demo() -> None:
    pl = _import_lightning()
    if pl is None:
        _run_fallback_pytorch()
        return

    print("\n" + "=" * 72)
    print("⚡ TORCH-AUDIT: LIGHTNING DEMO (current API)")
    print("=" * 72)

    # --- Minimal Lightning setup ---
    class LitModule(pl.LightningModule):  # type: ignore
        def __init__(self):
            super().__init__()
            self.model = TinyLitNet()
            self.loss_fn = nn.MSELoss()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ):
            x, y = batch
            pred = self(x)
            loss = self.loss_fn(pred, y)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
            return loss

        def configure_optimizers(self):
            # Intentionally "bad" optimizer config to trigger TA401/TA402.
            return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)

    # --- Torch-Audit callback implemented inside the demo ---
    class TorchAuditCallback(pl.Callback):  # type: ignore
        """Minimal callback that runs Torch-Audit checks.

        This pattern keeps Torch-Audit fully optional (no hard dependency inside torch_audit).
        """

        def __init__(self, every_n_steps: int = 1):
            super().__init__()
            self.every_n_steps = every_n_steps
            self.auditor: Auditor | None = None
            self._last_batch: Any = None

        def on_fit_start(self, trainer, pl_module) -> None:
            opt = (
                trainer.optimizers[0] if getattr(trainer, "optimizers", None) else None
            )
            self.auditor = Auditor(
                pl_module,
                optimizer=opt,
                every_n_steps=self.every_n_steps,
                reporters=[ConsoleReporter()],
            )
            self.auditor.attach()
            self.auditor.audit_static(step=getattr(trainer, "global_step", 0))
            self.auditor.audit_init(step=getattr(trainer, "global_step", 0))

        def on_train_batch_start(
            self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int = 0
        ):
            if self.auditor is None:
                return
            self._last_batch = batch
            self.auditor.step = int(getattr(trainer, "global_step", self.auditor.step))
            self.auditor.audit_data(batch, step=self.auditor.step)

        def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0,
        ):
            if self.auditor is None:
                return
            # After Lightning's training_step forward has run, forward hooks have data.
            self.auditor.step = int(getattr(trainer, "global_step", self.auditor.step))

            try:
                from torch_audit.validators.builtin.data import DataValidator

                post_validators = [
                    v
                    for v in self.auditor.validators
                    if not isinstance(v, DataValidator)
                ]
            except Exception:
                post_validators = self.auditor.validators

            ctx = self.auditor._make_context(
                Phase.FORWARD, step=self.auditor.step, batch=batch
            )
            if self.auditor._should_audit(ctx.step, ctx.phase):
                self.auditor.runner.run_step(ctx, validators=post_validators)

        def on_after_backward(self, trainer, pl_module) -> None:
            if self.auditor is None:
                return
            self.auditor.step = int(getattr(trainer, "global_step", self.auditor.step))
            ctx = self.auditor._make_context(
                Phase.BACKWARD, step=self.auditor.step, batch=self._last_batch
            )
            if self.auditor._should_audit(ctx.step, ctx.phase):
                self.auditor.runner.run_step(ctx)

        # Different Lightning versions expose different hooks. We implement both.
        def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:  # type: ignore
            # We prefer checking after the step, but not all versions have on_after_optimizer_step.
            pass

        def on_after_optimizer_step(self, trainer, pl_module, optimizer) -> None:  # type: ignore
            if self.auditor is None:
                return
            self.auditor.step = int(getattr(trainer, "global_step", self.auditor.step))
            ctx = self.auditor._make_context(
                Phase.OPTIMIZER, step=self.auditor.step, batch=self._last_batch
            )
            if self.auditor._should_audit(ctx.step, ctx.phase):
                self.auditor.runner.run_step(ctx)

        def on_fit_end(self, trainer, pl_module) -> None:
            if self.auditor is None:
                return
            try:
                self.auditor.finish(report=True)
            finally:
                self.auditor.detach()

    # Data: tiny batch to intentionally trigger TA304 with BatchNorm.
    dataset = TensorDataset(torch.randn(20, 32), torch.randn(20, 1))
    loader = DataLoader(dataset, batch_size=2)

    lit = LitModule()

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[TorchAuditCallback(every_n_steps=1)],
    )
    trainer.fit(lit, loader)


if __name__ == "__main__":
    run_demo()
