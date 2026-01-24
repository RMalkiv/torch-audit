# Rules

This is an auto-generated list of built-in **torch-audit** rules. To regenerate: `python scripts/generate_rules.py`

## Rule packs

- `TA1xx` — Stability
- `TA2xx` — Hardware
- `TA3xx` — Data integrity
- `TA4xx` — Architecture & optimization
- `TA5xx` — Runtime graph / hooks

## Index

| ID | Default severity | Category | Title |
|---:|:-----------------|:---------|:------|
| `TA000` | WARN | Internal | Internal Validator Error |
| `TA100` | ERROR | Stability | NaN or Inf Detected |
| `TA102` | WARN | Stability | Gradient Explosion |
| `TA103` | INFO | Stability | Dead Units (Zero Grads) |
| `TA104` | ERROR | Stability | No Gradients Found |
| `TA105` | WARN | Stability | Activation Collapse (Dead Neurons) |
| `TA200` | WARN | Performance | Tensor Core Alignment |
| `TA201` | WARN | Performance | Suboptimal Memory Layout |
| `TA202` | ERROR | Performance | Model Device Placement |
| `TA203` | INFO | Performance | Precision Check (AMP) |
| `TA300` | ERROR | Data Integrity | Input Device Mismatch |
| `TA301` | ERROR | Data Integrity | Suspicious Data Range |
| `TA302` | WARN | Data Integrity | Flat/Empty Data |
| `TA303` | WARN | Data Integrity | Suspicious Input Layout (NHWC vs NCHW) |
| `TA304` | WARN | Data Integrity | Tiny Batch Size |
| `TA305` | ERROR | Data Integrity | Invalid Input Values |
| `TA400` | WARN | Architecture | Redundant Bias before Norm |
| `TA401` | WARN | Optimization | AdamW vs Adam |
| `TA402` | WARN | Optimization | Weight Decay on Norm/Bias |
| `TA403` | WARN | Optimization | Weight Decay on Embeddings |
| `TA404` | INFO | Architecture | Even Kernel Size |
| `TA405` | WARN | Architecture | Dead Convolution Filters |
| `TA500` | ERROR | Architecture | Unused Layer (Zombie) |
| `TA501` | ERROR | Architecture | Stateful Layer Reuse |

## Details

### TA000 — Internal Validator Error

- **Category:** Internal
- **Default severity:** WARN

**Description**

A validator crashed during execution.

**Remediation**

Check stack trace.

### TA100 — NaN or Inf Detected

- **Category:** Stability
- **Default severity:** ERROR

**Description**

Parameters or gradients contain invalid values (NaN/Inf).

**Remediation**

Check learning rate, loss scaling, or initialization.

### TA102 — Gradient Explosion

- **Category:** Stability
- **Default severity:** WARN

**Description**

Global gradient norm is excessively high.

**Remediation**

Check learning rate or apply `torch.nn.utils.clip_grad_norm_`.

### TA103 — Dead Units (Zero Grads)

- **Category:** Stability
- **Default severity:** INFO

**Description**

Parameter gradients are entirely zero for the batch (Dead Neurons).

**Remediation**

Check initialization (e.g. ReLU dead units) or data flow.

### TA104 — No Gradients Found

- **Category:** Stability
- **Default severity:** ERROR

**Description**

No gradients were found in the model during backward pass.

**Remediation**

Ensure `loss.backward()` was called and parameters are not frozen.

### TA105 — Activation Collapse (Dead Neurons)

- **Category:** Stability
- **Default severity:** WARN

**Description**

A high percentage of neurons in a layer are outputting zero (Dead ReLU).

**Remediation**

Check initialization, lower learning rate, or switch activation function (e.g. LeakyReLU).

### TA200 — Tensor Core Alignment

- **Category:** Performance
- **Default severity:** WARN

**Description**

Dimensions should be divisible by 8 (FP16) or 16 (INT8) for maximum throughput.

**Remediation**

Pad dimensions to multiples of 8 (FP16) or 16 (INT8).

### TA201 — Suboptimal Memory Layout

- **Category:** Performance
- **Default severity:** WARN

**Description**

Conv2d/Conv3d layers are in NCHW/NCDHW format. Channels Last is faster on modern GPUs.

**Remediation**

Use `.to(memory_format=torch.channels_last)` or `channels_last_3d`.

### TA202 — Model Device Placement

- **Category:** Performance
- **Default severity:** ERROR

**Description**

Model is split across devices (Split Brain) or on CPU when GPU is available.

**Remediation**

Move all parameters to the correct device (e.g. `model.cuda()`).

### TA203 — Precision Check (AMP)

- **Category:** Performance
- **Default severity:** INFO

**Description**

Model seems to be using full FP32 precision. Modern GPUs run faster with AMP (FP16/BF16).

**Remediation**

Use `torch.amp.autocast()` or convert weights to `bfloat16`.

### TA300 — Input Device Mismatch

- **Category:** Data Integrity
- **Default severity:** ERROR

**Description**

Input batch is on a different device (e.g., CPU) than the model (GPU).

**Remediation**

Move data to the correct device using `.to(device, non_blocking=True)`.

### TA301 — Suspicious Data Range

- **Category:** Data Integrity
- **Default severity:** ERROR

**Description**

Input data range is suspicious (e.g., [0, 255] for float inputs).

**Remediation**

Normalize inputs to [0, 1] or [-1, 1].

### TA302 — Flat/Empty Data

- **Category:** Data Integrity
- **Default severity:** WARN

**Description**

Input batch has near-zero variance (blank images/tokens).

**Remediation**

Check data loader, augmentation pipeline, or file integrity.

### TA303 — Suspicious Input Layout (NHWC vs NCHW)

- **Category:** Data Integrity
- **Default severity:** WARN

**Description**

Input tensor shape looks like NHWC (Channels Last), but PyTorch expects NCHW.

**Remediation**

Ensure input is permuted correctly: `x.permute(0, 3, 1, 2)`.

### TA304 — Tiny Batch Size

- **Category:** Data Integrity
- **Default severity:** WARN

**Description**

Batch size is very small (e.g. < 8) while using BatchNorm. This causes training instability.

**Remediation**

Increase batch size or switch to GroupNorm/LayerNorm.

### TA305 — Invalid Input Values

- **Category:** Data Integrity
- **Default severity:** ERROR

**Description**

Input contains invalid values (e.g., negative integers for embeddings).

**Remediation**

Check data preprocessing and vocabulary mapping.

### TA400 — Redundant Bias before Norm

- **Category:** Architecture
- **Default severity:** WARN

**Description**

Linear/Conv layer has `bias=True` but is immediately followed by a Normalization layer.

**Remediation**

Set `bias=False` on the layer to save memory and compute.

### TA401 — AdamW vs Adam

- **Category:** Optimization
- **Default severity:** WARN

**Description**

Using `Adam` with `weight_decay > 0` is often inferior to `AdamW`.

**Remediation**

Switch to `torch.optim.AdamW` for decoupled weight decay.

### TA402 — Weight Decay on Norm/Bias

- **Category:** Optimization
- **Default severity:** WARN

**Description**

Weight decay is applied to Normalization layers or Bias terms.

**Remediation**

Set `weight_decay=0.0` for these parameter groups.

### TA403 — Weight Decay on Embeddings

- **Category:** Optimization
- **Default severity:** WARN

**Description**

Weight decay is applied to Embedding layers. This breaks sparse gradients and is often harmful.

**Remediation**

Set `weight_decay=0.0` for embeddings or use sparse optimizers.

### TA404 — Even Kernel Size

- **Category:** Architecture
- **Default severity:** INFO

**Description**

Convolution uses an even kernel size (e.g., 2, 4). This can cause aliasing or shift artifacts.

**Remediation**

Consider using odd kernel sizes (3, 5, 7) with symmetric padding.

### TA405 — Dead Convolution Filters

- **Category:** Architecture
- **Default severity:** WARN

**Description**

Some convolution filters have weights that are entirely zero (or near zero).

**Remediation**

Check initialization or pruning logic. These filters contribute nothing.

### TA500 — Unused Layer (Zombie)

- **Category:** Architecture
- **Default severity:** ERROR

**Description**

Layer is defined in the model but was never called during the forward pass.

**Remediation**

Remove the layer or set `find_unused_parameters=True` (DDP).

### TA501 — Stateful Layer Reuse

- **Category:** Architecture
- **Default severity:** ERROR

**Description**

A stateful layer (e.g. BatchNorm) was called multiple times.

**Remediation**

Use distinct layers for each pass to avoid statistics corruption.
