import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl

from torch_audit import Auditor
from torch_audit.callbacks import LightningAuditCallback


class SabotagedLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()

        # Issue: Zombie Layer (Defined but never used)
        self.ghost_layer = nn.Linear(128, 128)

        # Issue: Bad Initialization -> Dead Neurons
        # We force weights to be negative large numbers so ReLU outputs zeros
        with torch.no_grad():
            self.fc1.weight.fill_(-10.0)
            self.fc1.bias.fill_(-10.0)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def run_demo():
    print("\n" + "=" * 60)
    print("⚡ TORCH-AUDIT: LIGHTNING FAILURE DEMO")
    print("=" * 60)

    # 1. Setup Data
    inputs = torch.randn(100, 32)
    targets = torch.randn(100, 1)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=10, num_workers=0)

    # 2. Setup Model & Auditor
    model = SabotagedLightningModel()

    config = {
        'monitor_graph': True,  # Enable zombie check
        'monitor_dead_neurons': True,
        'interval': 1  # Audit every step for this short demo
    }
    auditor = Auditor(model, config=config)

    # 3. Trainer with Callback
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=3,
        enable_checkpointing=False,
        logger=False,
        # The Magic Line:
        callbacks=[LightningAuditCallback(auditor)]
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    run_demo()