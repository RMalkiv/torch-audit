import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from torch_audit import Auditor, AuditConfig
from torch_audit.callbacks import LightningAuditCallback


class SabotagedLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Issue: Zombie Layer
        self.ghost = nn.Linear(128, 128)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def run_demo():
    print("\n" + "=" * 60)
    print("⚡ TORCH-AUDIT: LIGHTNING DEMO")
    print("=" * 60)

    # 1. Setup Data with "Tiny Batch" issue (Batch size 2 < 8)
    dataset = TensorDataset(torch.randn(20, 32), torch.randn(20, 1))
    loader = DataLoader(dataset, batch_size=2) # Warning: Tiny Batch

    # 2. Setup Auditor
    model = SabotagedLightningModel()
    config = AuditConfig(monitor_graph=True, interval=1)
    auditor = Auditor(model, config=config)

    # 3. Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        enable_checkpointing=False,
        logger=False,
        callbacks=[LightningAuditCallback(auditor)]
    )
    trainer.fit(model, loader)

if __name__ == "__main__":
    run_demo()