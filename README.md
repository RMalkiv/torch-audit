# 🔥 torch-audit
### The Linter for PyTorch Models

[![PyPI](https://img.shields.io/pypi/v/torch-audit)](https://pypi.org/project/torch-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**torch-audit** is a "check engine light" for your Deep Learning training loop. It detects silent bugs that don't crash your code but ruin your training.

- 💥 **Exploding Gradients:** Detects norm spikes.
- 🧟 **Zombie Layers:** Identifies DDP-unsafe layers that are defined but never used.
- 🐌 **Hardware Waste:** Warns about shapes incompatible with Tensor Cores.
- 🧠 **Domain Awareness:** Specific modules for **NLP** (Tokenization waste) and **CV** (Dead filters).
- 📉 **Dead Neurons:** Finds layers that output large number of zeros (ReLU death).

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

## 🚀 Quick Start (Zero Overhead)
Wrap your training step with auditor.audit_dynamic(). By default, it runs every step, but you can schedule it to run once every 1000 steps for zero performance penalty.


```python
import torch
from torch_audit import Auditor

# 1. Setup Auditor (Audits 1 step, sleeps for 999)
auditor = Auditor(model, config={'interval': 1000})

# 2. Static Audit (Run once before training)
# Checks architecture, unused layers, and weight initialization
auditor.audit_static()

# 3. Training Loop
for batch in dataloader:
    # The Context Manager:
    # - If active: Hooks attached, full analysis running.
    # - If sleeping: Zero overhead (literally just a generic yield).
    with auditor.audit_dynamic():
        pred = model(batch)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
```
### The Output
When a bug is found, torch-audit prints a beautiful report to your console:

```text
🚀 Audit Running (Step 5000)...
                            ⚠️ Audit Report (Step 5000)                            
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Type              ┃ Layer         ┃ Message                                     ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 🔴 DDP Safety     │ ghost_layer   │ Layer defined but NEVER called (Zombie).    │
│ 🟡 Activations    │ layer4.relu   │ 95.5% zeros (Dead Neurons).                 │
│ 🟡 Hardware       │ fc1           │ Dimensions (127->64) not divisible by 8.    │
└───────────────────┴───────────────┴─────────────────────────────────────────────┘
```

## 🧩 Integrations
We support the ecosystem you already use.

### ⚡ PyTorch Lightning
Zero code changes to your loop. Just add the callback.
```python
from lightning.pytorch import Trainer
from torch_audit import Auditor
from torch_audit.callbacks import LightningAuditCallback

auditor = Auditor(model, config={'interval': 100})
trainer = Trainer(callbacks=[LightningAuditCallback(auditor)])
```

### 🤗 Hugging Face Trainer
Plug-and-play with the Trainer API.
```python
from transformers import Trainer
from torch_audit import Auditor
from torch_audit.callbacks import HFAuditCallback

auditor = Auditor(model, config={'monitor_nlp': True})
trainer = Trainer(..., callbacks=[HFAuditCallback(auditor)])
```

## 🧠 Domain Specific Audits
torch-audit has even more tools for your domain.

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
Catches:

    ⚠️ "High Padding detected (55% of tokens). 50%+ of compute is wasted."

    🔴 "High [UNK] rate (8%). Tokenizer mismatch."

    ℹ️ "Embedding and Output Head are not tied."

### 🖼️ Computer Vision Mode
Detects normalization bugs (0-255 inputs) and dead convolution filters.
```python
auditor = Auditor(model, config={'monitor_cv': True})
```
Catches:

    🔴 "Input values range [0, 255]. Did you forget ToTensor()?"

    🟡 "Layer conv2 has 45% dead filters (magnitude ~0)."

## ⚙️ Configuration

You can configure the auditor via a dictionary or the `AuditConfig` object.

| Parameter | Default | Description                                                           |
| :--- | :--- |:----------------------------------------------------------------------|
| `interval` | `1` | Run audit every N steps. Set to `100`, `1000` or more for production. |
| `limit` | `None` | Stop auditing after N reports.                                        |
| `float_threshold` | `10.0` | Max value allowed in inputs before warning.                           |
| `monitor_dead_neurons` | `True` | Check for activations death.                                          |
| `monitor_graph` | `True` | Check for unused (zombie) layers.                                     |
| `monitor_nlp` | `False` | Enable NLP-specific hooks (requires `pad_token_id`).                  |
| `monitor_cv` | `False` | Enable CV-specific hooks.                                             |

## 🛠️ Advanced: Manual Triggering

Sometimes you want to audit, for example, when the loss spikes.
```python
loss = criterion(output, target)

if loss.item() > 10.0:
    print("Loss spike! Debugging next step...")
    auditor.schedule_next_step() # Forces audit on next forward pass
```

## License

Distributed under the MIT License.
