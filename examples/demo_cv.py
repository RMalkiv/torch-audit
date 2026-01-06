import torch
import torch.nn as nn
from torch_audit import Auditor


class BadConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Issue 1: Kernel size 2x2
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 31 * 31, 10)

    def forward(self, x):
        return self.fc(self.flatten(self.relu(self.conv1(x))))


def run_demo():
    print("\n" + "=" * 60)
    print("🖼️ TORCH-AUDIT: COMPUTER VISION DEMO")
    print("=" * 60)

    model = BadConvNet()

    # Issue: Dead Filters
    # We manually zero out 80% of the convolution filters
    with torch.no_grad():
        model.conv1.weight.data[5:] = 0.0

    auditor = Auditor(model, config={'monitor_cv': True})

    # 1. Static Audit (Checks Dead Filters & Kernels)
    auditor.audit_static()

    print("\n[Simulating Bad Preprocessing]...")

    # Issue: Raw Images [0, 255] passed without normalization
    raw_images = torch.randint(0, 256, (4, 3, 32, 32)).float()

    with auditor.audit_dynamic():
        # The auditor handles the logic to verify data inside audit_data
        # or via manual inspection if you call audit_data explicitly.
        auditor.audit_data(raw_images)
        model(raw_images)


if __name__ == "__main__":
    run_demo()