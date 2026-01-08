import torch
import torch.nn as nn
from torch_audit import Auditor, AuditConfig


class BadConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Issue 1: Kernel 2x2 (Even size alignment)
            # Issue 2: Bias=True followed by BN (Redundant parameters)
            nn.Conv2d(3, 16, kernel_size=2, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 31 * 31, 10)

    def forward(self, x):
        return self.fc(self.flatten(self.features(x)))


def run_demo():
    print("\n" + "=" * 60)
    print("🖼️ TORCH-AUDIT: COMPUTER VISION DEMO")
    print("=" * 60)

    model = BadConvNet()

    # Issue 3: Dead Filters (80% pruned)
    with torch.no_grad():
        model.features[0].weight.data[3:] = 0.0

    auditor = Auditor(model, config=AuditConfig(monitor_cv=True))

    # 1. Static Audit
    auditor.audit_static()

    print("\n[Simulating Data Issues]...")

    # Issue 4: Wrong Layout [Batch, Height, Width, Channel] (NHWC)
    # PyTorch expects [Batch, Channel, Height, Width]
    bad_layout_img = torch.randn(4, 32, 32, 3)

    with auditor.audit_dynamic():
        # This will trigger 'CV Data Layout' error
        auditor.audit_data(bad_layout_img)

        # We don't run forward() because the shape would crash PyTorch
        # But the auditor catches it BEFORE the crash!


if __name__ == "__main__":
    run_demo()
