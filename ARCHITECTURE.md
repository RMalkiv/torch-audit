# torch-audit architecture

This document explains the high-level structure of **torch-audit** and how the pieces fit together.

## Design goals

- **Training-loop safe**: auditing should never crash a training run.
- **Runtime truth**: prefer what actually happens (real tensors, real grads) over static guesses.
- **Low friction**: easy to adopt in existing loops.
- **Composable**: users can add custom validators and reporters.

## Mental model

Think of torch-audit as:

```
(model + optimizer + optional batch)
           │
           ▼
      AuditContext (phase, step, model, optimizer, batch)
           │
           ▼
   validators.check(context)  ───▶ Finding(rule_id, message, severity, metadata)
           │
           ▼
        AuditRunner
           │
           ▼
        AuditResult  ───▶ reporters (rich console / json / sarif)
```

## Core concepts

### Phases

Audits run in one of several phases:

- `static` — model structure (architecture hints)
- `init` — optimizer configuration
- `forward` — post-forward runtime checks (hooks can observe activations)
- `backward` — gradient checks
- `optimizer` — post-optimizer-step checks

### Validators

A validator is a small unit of logic that inspects an `AuditContext` and yields `Finding` objects.

Validators may be:

- **stateless**: purely inspect the context (fast, simple)
- **stateful**: attach hooks (e.g. graph/activation checks)

Stateful validators can override:

- `attach(model)` / `detach()`
- `on_phase_start(ctx)` / `on_phase_end(ctx)`

torch-audit treats these lifecycle methods as **best-effort** so instrumentation failures cannot break training.

### Runner

`AuditRunner` is responsible for:

- executing validators for a phase
- enforcing the contract that validators only emit declared `Rule` IDs
- applying `--select/--ignore` filtering
- applying suppressions
- baseline gating ("new" vs "all" findings)
- producing the final `AuditResult`

### Reporters

Reporters are pure output formatters:

- `ConsoleReporter` — human-friendly Rich output
- `JSONReporter` — machine readable output
- `SARIFReporter` — GitHub code scanning compatible output

## Extension points

### Custom validators

Create a new validator by inheriting from `BaseValidator` and adding it to the validator list passed to `Auditor(...)` or `audit(...)`.

### Custom reporters

Implement the `Reporter` interface and pass it to `Auditor(..., reporters=[...])`.

## Runtime integrations

torch-audit supports multiple integration styles:

- **Explicit wrappers**: `auditor.forward(...)`, `auditor.backward(loss)`, `auditor.optimizer_step()`
- **Decorator**: `@audit_step(auditor)` for post-step auditing
- **Zero-touch**: `autopatch(...)` monkey patches `model.forward` and/or `optimizer.step`

Prefer explicit wrappers for maximum transparency and accurate phase attribution.
