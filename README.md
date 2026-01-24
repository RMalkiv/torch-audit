# 🔥 torch-audit
### Runtime auditing for PyTorch training loops

[![PyPI](https://img.shields.io/pypi/v/torch-audit?cache=none)](https://pypi.org/project/torch-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/RMalkiv/torch-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/RMalkiv/torch-audit/actions/workflows/ci.yml)

**torch-audit** is a “check engine light” for your **training loop**.

Unlike a static linter, **torch-audit runs at runtime** and inspects what actually happens during training:

- real tensors and batches (device placement, suspicious ranges, layouts)
- real optimizer configuration (weight decay pitfalls)
- real gradients (NaNs/Infs, explosions, missing grads)
- real model execution (unused “zombie” layers, stateful layer reuse)

The goal is to catch **silent bugs** that don’t crash your code but quietly ruin training or waste compute.

---

## 📦 Installation

```bash
pip install torch-audit
```

To run the optional integration demos you may also want:

```bash
pip install lightning transformers accelerate
```

---

## 🚀 Quick Start

### Zero-touch mode: `autopatch()`

If you want the **least code churn**, use `autopatch()`.
It monkey-patches **`model.forward`** and **`optimizer.step`** so a *normal* training loop automatically emits findings.

```python
import torch
from torch_audit import autopatch
from torch_audit.reporters.console import ConsoleReporter

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Audits every 1000 optimizer steps (set to 1 to audit every step)
auditor = autopatch(
    model,
    optimizer=optimizer,
    every_n_steps=1000,
    reporters=[ConsoleReporter()],
    fail_level="ERROR",
    run_static=True,
    run_init=True,
)

for batch, targets in dataloader:
    batch = batch.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    # ✅ No wrappers required
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Finalize results (and report again if you want)
result = auditor.finish(report=True)

# Restore original methods / detach hooks
# (recommended if you keep using the model after auditing)
auditor.unpatch()
```

> Note: `autopatch()` modifies objects in-place. If you rely on compilation/tracing tools
> (e.g. `torch.compile`, TorchScript), prefer the explicit wrapper mode below.

### Wrapper mode: `audit_dynamic(...)` + phase wrappers

If you want the **most accurate phase reporting** (forward/backward/optimizer) and the clearest control,
wrap your loop with `audit_dynamic(...)` and call the wrappers.

```python
import torch
from torch_audit import audit_dynamic
from torch_audit.reporters.console import ConsoleReporter

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Audits every 1000 optimizer steps (set to 1 to audit every step)
with audit_dynamic(
    model,
    optimizer=optimizer,
    every_n_steps=1000,
    reporters=[ConsoleReporter()],
    fail_level="ERROR",
) as auditor:
    for batch, targets in dataloader:
        batch = batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Use the wrappers so audits run in the right phase.
        outputs = auditor.forward(batch)
        loss = criterion(outputs, targets)
        auditor.backward(loss)
        auditor.optimizer_step()

# Finalize results (and report again if you want)
result = auditor.finish(report=False)
```

### A lighter-weight option: `audit_step()`

If you already have a training-step function and want a **minimal, opt-in** integration,
you can use the decorator. It runs a best-effort **post-step** audit (optimizer phase).

```python
from torch_audit import Auditor, audit_step
from torch_audit.reporters.console import ConsoleReporter

auditor = Auditor(model, optimizer=optimizer, every_n_steps=1000, reporters=[ConsoleReporter()])

@audit_step(auditor)
def train_step(batch, targets):
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    return loss

with auditor:
    auditor.audit_static()
    auditor.audit_init()
    for batch, targets in dataloader:
        train_step(batch, targets)
```

If you want the most accurate runtime results (graph/activation checks, precise phase reporting),
prefer the explicit wrappers: `auditor.forward()`, `auditor.backward()`, `auditor.optimizer_step()`.

---

## 📂 Runnable demos

The `examples/` folder contains runnable scripts designed to trigger findings.

- `python examples/demo_general.py` — plain PyTorch loop, end-to-end runtime auditing
- `python examples/demo_cv.py` — CV-ish model + common data/layout mistakes
- `python examples/demo_nlp.py` — “NLP-ish” tensors (e.g., invalid token ids) + optimizer pitfalls
- `python examples/demo_lightning.py` — Lightning integration pattern (demo includes a minimal callback)
- `python examples/demo_hf.py` — Transformers pattern (no downloads; constructs a tiny model from config)
- `python examples/demo_accelerate.py` — Accelerate pattern (audits around `accelerator.backward(loss)`)

> Note: the repository currently focuses on the core runtime engine. Some ecosystem integrations are shown as **copy-paste patterns** in the demos rather than shipped as a first-class API.

---

## 📚 Reference

- [Rule reference](RULES.md) — all rule IDs, titles, severities, and remediation
- [Architecture](ARCHITECTURE.md) — how the runtime engine is structured

---

## 🧰 One-shot audits (CLI / CI)

You can run an audit without a training loop (useful for CI smoke checks).

> Tip: if the `torch-audit` command isn’t available in your environment, run the same command as:
> `python -m torch_audit ...`

```bash
# List all available rules
torch-audit --list-rules

# Explain a single rule (ID)
torch-audit --explain TA405

# Static checks (architecture / hardware hints)
torch-audit my_project.models:MyModel --phase static

# Init checks (optimizer config, weight decay pitfalls)
torch-audit my_project.models:MyModel --phase init

# JSON output (machine readable)
torch-audit my_project.models:MyModel --phase static -f json -o audit.json

# SARIF output (GitHub code scanning / security tab)
torch-audit my_project.models:MyModel --phase static -f sarif -o audit.sarif
```

### Baselines and rule filtering

```bash
# Create / update a baseline file from the current findings
torch-audit my_project.models:MyModel --phase static --baseline baseline.json --update-baseline

# Only fail on new findings compared to the baseline
torch-audit my_project.models:MyModel --phase static --baseline baseline.json

# Run only specific rules
torch-audit my_project.models:MyModel --phase static --select TA200,TA202

# Ignore specific rules
torch-audit my_project.models:MyModel --phase static --ignore TA201
```

---

## 🛠️ What it checks today

This repo currently ships the following built-in validators:

### Data integrity (runtime)
- **TA300** input device mismatch (e.g., CPU batch with GPU model)
- **TA301** suspicious float ranges (e.g., normalized data missing)
- **TA302** flat/empty tensors (near-zero variance)
- **TA303** suspicious layout heuristic (NHWC vs NCHW)
- **TA304** tiny batch sizes with BatchNorm
- **TA305** invalid integer inputs (e.g., negative token ids for embeddings)

### Stability (runtime)
- **TA100** NaNs/Infs in parameters or gradients
- **TA102** gradient explosion (global grad norm)
- **TA103** “dead units” (exactly zero grads)
- **TA104** no gradients found
- **TA105** activation collapse / high sparsity (forward hooks)

### Optimization config (static/init)
- **TA401** Adam + weight_decay (suggest AdamW)
- **TA402** weight decay applied to norm/bias params
- **TA403** weight decay applied to embeddings

### Architecture + execution (static + runtime)
- **TA400** redundant bias before normalization
- **TA404** even convolution kernel sizes
- **TA405** dead convolution filters
- **TA500** unused “zombie” layers (runtime, forward hooks)
- **TA501** stateful layer reuse (e.g., BatchNorm called multiple times)

### Hardware/performance hints (static/init)
- **TA200** tensor-core alignment hints
- **TA201** channels-last memory layout hints
- **TA202** model device placement / split-brain
- **TA203** AMP/precision suggestion

---

## 🧾 Reporters

You can output results to multiple formats:

```python
from torch_audit.runtime import Auditor
from torch_audit.reporters.console import ConsoleReporter
from torch_audit.reporters.json import JSONReporter
from torch_audit.reporters.sarif import SARIFReporter

auditor = Auditor(
    model,
    optimizer=optimizer,
    reporters=[
        ConsoleReporter(),
        JSONReporter(dest="audit.json"),
        SARIFReporter(dest="audit.sarif"),
    ],
)
```

---

## 🤝 Contributing & feedback

If you find a silent bug torch-audit missed, or want a new runtime validator,
please open an issue.

## License

Distributed under the MIT License.
