import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

from torch_audit import Auditor, AuditConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)


def run_demo():
    # 1. Initialize Accelerator
    accelerator = Accelerator()

    # Only print header on the main process to avoid log spam
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("🚀 TORCH-AUDIT: ACCELERATE (DDP) DEMO")
        print("=" * 60)

    # 2. Setup Components
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(64, 10)
    targets = torch.randn(64, 1)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=8)

    # 3. Prepare with Accelerator
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # 4. Setup Auditor
    # Auditor automatically handles DDP:
    # - It attaches hooks on ALL processes (required for correct DDP sync)
    # - It prints reports ONLY on Rank 0 (to keep logs clean)
    config = AuditConfig(interval=2)
    auditor = Auditor(model, config=config)

    model.train()

    if accelerator.is_main_process:
        print(f"\n[Running on {accelerator.num_processes} devices]...")

    for epoch in range(2):
        for step, batch in enumerate(dataloader):
            # The Context Manager works seamlessly with Accelerate
            with auditor.audit_dynamic():
                x, y = batch
                outputs = model(x)
                loss = nn.functional.mse_loss(outputs, y)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

    if accelerator.is_main_process:
        print("\n[Done]")


if __name__ == "__main__":
    run_demo()