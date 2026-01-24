# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**, and this project follows **Semantic Versioning**.

## [0.3.0] - 2026-01-24

This release is a major usability + DevEx upgrade and includes several **breaking API/config changes**
relative to the GitHub `v0.2.0` tag (and the `main` README at the time).

### Added

- **Baselines**
  - `--baseline <file>.json` support to treat existing findings as “known” and fail only on **new** findings.
  - `--update-baseline` to (re)write the baseline file from the current run.
  - Stable finding **fingerprints** to make baseline diffs deterministic.

- **Rule filtering + suppressions**
  - `--select TAxxx` / `--ignore TAxxx` filtering.
  - Config-driven suppressions with optional module regex matching.
  - `--show-suppressed` option to include suppressed findings in output (but not in failure evaluation).

- **One-shot auditing CLI**
  - `torch-audit <module.path:ModelOrFactory>` for CI-friendly audits.
  - `--list-rules` to print all rule IDs.
  - `--explain <RULE_ID>` to print full rule detail (description + remediation).
  - Multi-format output via `--format rich|json|sarif`, with file output for machine formats.

- **New reporters**
  - JSON reporter.
  - SARIF 2.1.0 reporter with deterministic ordering (stable diffs in CI systems).

- **Repo “masterpiece” polish**
  - GitHub Actions workflows (CI + release).
  - Issue templates + PR template.
  - CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT, ARCHITECTURE, RULES docs.
  - Pre-commit configuration, editorconfig/gitattributes, and a Makefile.
  - A proper test suite skeleton.

### Changed (Breaking)

- **Python support**
  - Target/runtime baseline moved to **Python 3.10+** (project tooling targets `py310`).

- **Config model**
  - The “wide” `AuditConfig(...)` surface documented in v0.2.0 is replaced by a smaller,
    rule-first configuration:
    - `fail_level`, `baseline_file`, `update_baseline`
    - `select_rules`, `ignore_rules`, `show_suppressed`
    - `suppress_internal_errors`, `suppressions=[...]`
  - v0.2.0-era knobs like `interval`, `monitor_nlp`, `monitor_cv`, etc. are no longer the primary interface.
    (Rule selection + validators are now the intended mechanism.)

- **Runtime API shape**
  - The primary scheduling knob is now `every_n_steps` on the runtime `Auditor`.
  - Helper entry points are oriented around:
    - `autopatch(model, optimizer=...)` (zero-touch integration)
    - `audit_dynamic(model, optimizer=...)` (context manager creating an `Auditor`)
    - `audit_step(auditor)(fn)` (decorator factory)
  - Code that relied on `@auditor.audit_step` or `with auditor.audit_dynamic():` must migrate
    to the standalone helpers or explicit wrappers.

- **Documentation**
  - README refocused around:
    - `autopatch()` (drop-in training loop auditing)
    - wrapper-based auditing for explicit phases
    - CLI/CI usage + baselines/rule filtering
  - New RULES.md as a single reference point for rule IDs and intent.

### Fixed

- **Deterministic outputs**
  - SARIF/JSON output is now stable across runs (sorting + stable fingerprints),
    making it suitable for CI comparisons and baselines.

- **Runtime robustness**
  - Validator crashes are captured as an internal finding (TA000) instead of crashing the user’s run
    (unless `suppress_internal_errors=True`).

- **Lint/quality**
  - Warnings emitted by baseline loading use an explicit `stacklevel` so users see the correct call site.

### Removed / Deprecated

- **Docs-first “integration surface”**
  - The “Lightning/HF callback integration” surface described in the v0.2.0 README is not part of the
    core 0.3.0 runtime API package layout in this refactor. (Extras remain reserved, but the integration
    code is expected to be reintroduced/reshaped explicitly.)

- **Manual scheduling API**
  - The v0.2.0 README’s `schedule_next_step()` style manual trigger is not present in the 0.3.0 runtime API.
    (If needed, reintroduce as a small `auditor.force_next()` convenience in a follow-up.)

---

## [0.2.0] - 2026-01-08

- GitHub-tagged baseline release documented on the repository with:
  - `Auditor` + `AuditConfig(interval=...)`
  - decorator usage via `@auditor.audit_step`
  - runtime context manager via `with auditor.audit_dynamic():`
  - documented (optional) ecosystem integrations and expanded config surface.

[0.3.0]: https://github.com/RMalkiv/torch-audit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/RMalkiv/torch-audit/tree/v0.2.0
