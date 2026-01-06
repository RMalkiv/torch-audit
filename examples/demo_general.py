import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming the package is installed or in path
from torch_audit import Auditor


class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Issue 1: Dimension 127 is not divisible by 8 (Tensor Core Warning)
        self.fc1 = nn.Linear(64, 127)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(127, 10)

        # Issue 2: Zombie Layer (Defined but never used in forward)
        self.ghost_layer = nn.Linear(128, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def run_demo():
    print("\n" + "=" * 60)
    print("🔥 TORCH-AUDIT: GENERAL FAILURE DEMO")
    print("=" * 60)

    model = BrokenModel()

    # Issue 3: Bad Initialization (Causes Dead Neurons)
    # We force weights negative so ReLU outputs strict zeros
    with torch.no_grad():
        model.fc1.weight.fill_(-1.0)
        model.fc1.bias.fill_(-1.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    auditor = Auditor(model, optimizer, config={'monitor_graph': True})

    # 1. Static Audit (Architecture & Weights)
    auditor.audit_static()

    # Create dummy data
    inputs = torch.randn(4, 64)
    targets = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Running Training Step]...")

    # 2. Dynamic Audit (Runtime)
    # Using the new Context Manager pattern
    with auditor.audit_dynamic():
        # A. Data Hygiene (Manual call, or integrate into loop)
        auditor.audit_data(inputs)

        # B. Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # C. Backward (Gradient Checks happen here)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    run_demo()