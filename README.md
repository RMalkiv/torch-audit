# 🔥 torch-audit
### The Linter for PyTorch Models

[![PyPI](https://img.shields.io/pypi/v/torch-audit?cache=none)](https://pypi.org/project/torch-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**torch-audit** is a "check engine light" for your Deep Learning training loop. It detects silent bugs that don't crash your code but ruin your training or waste compute.

- 🖥️ **Hardware Efficiency:** Detects slow memory layouts (NHWC vs NCHW), mixed-precision failures, and tensor core misalignment.
- 🧪 **Data Integrity:** Catches broken attention masks, CV layout bugs, and silent NaN/Inf propagation.
- 📉 **Training Stability:** Identifies exploding gradients, bad optimizer config (Adam vs AdamW), and "dead" neurons.
- 🧟 **Graph Logic:** Identifies DDP-unsafe "Zombie" layers and redundant computations (e.g., Bias before BatchNorm).
- 🧠 **Domain Awareness:** Deep inspection for **NLP** (Padding waste, Tokenizer quality) and **CV** (Dead filters, Redundant biases).

---

## 📦 Installation

Install the standard version (lightweight):
```bash
pip install torch-audit
```

### Optional Integrations:
```
# For PyTorch Lightning support
pip install "torch-audit[lightning]"

# For Hugging Face Transformers support
pip install "torch-audit[hf]"

# For everything
pip install "torch-audit[all]"
```

## 🚀 Quick Start
You have two ways to use `torch-audit`: the **Decorator** (easiest) or the **Context Manager** (most control).

### The Decorator Method (Recommended)
```python
import torch
from torch_audit import Auditor, AuditConfig

# 1. Setup Auditor (Audits every 1000 steps)
config = AuditConfig(interval=1000)
auditor = Auditor(model, optimizer, config=config)

# 2. Static Audit (Run once before training)
# Checks architecture, unused layers, and weight initialization
auditor.audit_static()

# 3. Training Loop
# The decorator handles hooks, data auditing, and error reporting automatically.
@auditor.audit_step
def train_step(batch, targets):
    optimizer.zero_grad()
    pred = model(batch)
    loss = criterion(pred, targets)
    loss.backward()
    optimizer.step()

for batch, targets in dataloader:
    train_step(batch, targets)
```
### The Context Manager Method  
```python
# 3. Training Loop
for batch in dataloader:
    # Manual data check (optional but recommended)
    auditor.audit_data(batch)

    # Dynamic checks (Gradients, Activations, Stability)
    with auditor.audit_dynamic():
        pred = model(batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
```
### The Output
When a bug is found, `torch-audit` prints a structured report. It supports **Rich Console** tables (default) or **JSON/System Logs** for production.

```text
🚀 Audit Running (Step 5000)...
   🟡 Batch size is tiny (4). BatchNorm is unstable. (in Input Batch)

                            ⚠️ Audit Report (Step 5000)                            
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Type              ┃ Layer         ┃ Message                                     ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 🔴 DDP Safety     │ ghost_layer   │ Layer defined but NEVER called (Zombie).    │
│ 🔴 Data Integrity │ Input Batch   │ Attention Mask mismatch on 50 tokens.       │
│ 🟡 Tensor Core    │ fc1           │ Dims (127->64) not divisible by 8.          │
│ 🟡 Stability      │ Global        │ Optimizer epsilon (1e-08) too low for AMP.  │
│ 🔵 CV Opt         │ conv1         │ Bias=True followed by BatchNorm (Redundant).│
└───────────────────┴───────────────┴─────────────────────────────────────────────┘
```
## 📂 Runnable Demos
Don't just take our word for it! Break things yourself! We have prepared sabotaged scripts that trigger auditor warnings.

Check out the `examples/` folder:
- `python examples/demo_general.py` (General hardware/optimizer issues)
- `python examples/demo_nlp.py` (NLP & Tokenizer bugs)
- `python examples/demo_cv.py` (Computer Vision bugs)
- `python examples/demo_lightning.py` (PyTorch Lightning integration)
- `python examples/demo_hf.py` (Hugging Face integration)
- `python examples/demo_accelerate.py` (Accelerate integration)


## 🧩 Integrations
We support the ecosystem you already use.

### ⚡ PyTorch Lightning
Zero code changes to your loop. Just add the callback.
```python
from lightning.pytorch import Trainer
from torch_audit import Auditor, AuditConfig
from torch_audit.callbacks import LightningAuditCallback

auditor = Auditor(model, config=AuditConfig(interval=100))
trainer = Trainer(callbacks=[LightningAuditCallback(auditor)])
```

### 🤗 Hugging Face Trainer
Plug-and-play with the Trainer API.
```python
from transformers import Trainer
from torch_audit import Auditor, AuditConfig
from torch_audit.callbacks import HFAuditCallback

config = AuditConfig(monitor_nlp=True, interval=500)
auditor = Auditor(model, config=config)

trainer = Trainer(..., callbacks=[HFAuditCallback(auditor)])
```

## 🛠️ Capabilities & Modules
### 🖥️ Hardware & System (Always Active)

* **Device Placement:** Detects "Split Brain" (CPU/GPU mix) and forgotten `.cuda()` calls.
* **Tensor Cores:** Warns if matrix multiplications aren't aligned to 8 (FP16) or 16 (INT8).
* **Memory Layout:** Detects `NCHW` vs `NHWC` memory format issues.
* **Precision:** Suggests AMP/BFloat16 if model is 100% FP32.

### 🧪 Optimization & Stability

* **Config:** Warns if using `Adam` with `weight_decay` (suggests `AdamW`).
* **Regularization:** Detects weight decay applied to Biases or Norm layers.
* **Dynamics:** Checks for low `epsilon` in Mixed Precision (underflow risk).

### 📖 NLP Mode
Detects tokenizer issues, padding waste, and untied embeddings.
```python
config = {
    'monitor_nlp': True,
    'pad_token_id': tokenizer.pad_token_id, 
    'vocab_size': tokenizer.vocab_size
}
auditor = Auditor(model, config=config)
```

* **Data Integrity:** Checks if `attention_mask` actually masks the padding tokens in `input_ids`.
* **Efficiency:** Calculates wasted compute due to excessive padding (>50%).
* **Architecture:** Checks if Embedding weights are tied to the Output Head.

### 🖼️ Computer Vision Mode
Detects normalization bugs (0-255 inputs) and dead convolution filters.
```python
auditor = Auditor(model, config={'monitor_cv': True})
```
* **Layout:** Detects accidental `[Batch, Height, Width, Channel]` input (crashes PyTorch).
* **Redundant Bias:** Detects `Conv2d(bias=True)` followed immediately by `BatchNorm`.
* **Dead Filters:** Identifies convolution filters that have been pruned or collapsed to zero.

## ⚙️ Configuration

You can configure the auditor via a dictionary or the `AuditConfig` object.

| Parameter | Default | Description                                                     |
| :--- |:--------|:----------------------------------------------------------------|
| `interval` | `1`     | Run audit every N steps. Set to `1000+` or more for production. |
| `limit` | `None`  | Stop auditing after N reports.                                  |
| `float_threshold` | `10.0`  | Max value allowed in inputs before warning.                     |
| `monitor_dead_neurons` | `True`  | Check for activations death.                                    |
| `graph_atomic_modules` | `[]`    |List of custom layers (e.g. FlashAttn) to treat as leaves. 
| `monitor_graph` | `True`  | Check for unused (zombie) layers.                               |
| `monitor_nlp` | `False` | Enable NLP-specific hooks (requires `pad_token_id`).            |
| `monitor_cv` | `False` | Enable CV-specific hooks.                                       |

## 🏭 Production Logging
For headless training where you can't see the console, switch to the `LogReporter`.

```python
from torch_audit.core_old.reporter import LogReporter

# Writes to standard Python logging (INFO/WARN/ERROR)
auditor = Auditor(model, reporters=[LogReporter()])
```
## 🛠️ Manual Triggering

Sometimes you want to audit, for example, when the loss spikes.
```python
loss = criterion(output, target)

if loss.item() > 10.0:
    print("Loss spike! Debugging next step...")
    auditor.schedule_next_step() # Forces audit on next forward pass
```
## 🤝 Contributing & Feedback
Found a silent bug that `torch-audit` missed? Have a suggestion for a new Validator?
**[Open an Issue](https://github.com/RMalkiv/torch-audit/issues)!** We love feedback and contributions.

## License

Distributed under the MIT License.
