import torch
import torch.nn as nn
import torch.optim as optim
from torch_audit import Auditor, AuditConfig
from torch_audit.core_old.reporter import LogReporter, RichConsoleReporter


class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Issue 1: Dimension 127 (Not aligned to 8)
        self.fc1 = nn.Linear(64, 127)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(127, 10)
        # Issue 2: Zombie Layer
        self.ghost = nn.Linear(128, 128)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def run_demo():
    print("\n" + "=" * 60)
    print("🔥 TORCH-AUDIT: GENERAL DEMO")
    print("=" * 60)

    model = BrokenModel()

    # Issue 3: Bad Initialization (Dead Neurons)
    with torch.no_grad():
        model.fc1.weight.fill_(-1.0)
        model.fc1.bias.fill_(-1.0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 1. Setup Configuration
    config = AuditConfig(
        monitor_graph=True,
        monitor_dead_neurons=True,
        interval=1
    )

    # 2. Initialize Auditor
    # You can pass multiple reporters. Here we show both Console and standard Logging.
    auditor = Auditor(
        model,
        optimizer,
        config=config,
        reporters=[RichConsoleReporter(), LogReporter()]
    )

    # 3. Static Audit
    auditor.audit_static()

    inputs = torch.randn(4, 64)
    targets = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Scenario A: Using Context Manager]...")

    # --- METHOD A: Context Manager ---
    # Good for granular control inside loops
    with auditor.audit_dynamic():
        auditor.audit_data(inputs)  # Manual data check

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"\n[Scenario B: Using Decorator]...")

    # --- METHOD B: Decorator ---
    # Cleanest way for function-based steps.
    # Note: It automatically calls audit_data(inputs) if inputs is the first arg!
    @auditor.audit_step
    def train_step(batch_x, batch_y):
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

    # Just call the function normally
    train_step(inputs, targets)


if __name__ == "__main__":
    run_demo()