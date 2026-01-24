# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-23

### Added
- **Rule Packs**: Introduced 5 specialized validator packs with ~15 new rules:
  - `TA1xx`: Stability (NaNs, gradient explosion, dead units).
  - `TA2xx`: Hardware (split brain, tensor core alignment).
  - `TA3xx`: Data integrity (device mismatch, suspicious ranges).
  - `TA4xx`: Architecture & optimization (redundant bias, even kernel, weight decay targets).
  - `TA5xx`: Runtime graph (unused layer / zombie detection, stateful reuse).
- **CLI**: New `torch-audit` command line tool for one-shot audits.
- **Reporting**: SARIF output (`--format sarif`) for GitHub Code Scanning.
- **Reporting**: JSON output (`--format json`) for CI pipelines.
- **Architecture**: `batch` and `optimizer` support in `AuditContext`.

### Changed
- **Core**: `AuditResult` now tracks `new_findings` vs `all_findings` (baseline gating).
- **Core**: Fingerprints are strictly versioned (`v1:...`) for stable baselines.
- **Config**: Suppression regexes fail fast on invalid syntax.
- **Runtime**: Default `every_n_steps` is now safer for training loops.

### Fixed
- **Runtime safety**: Validator lifecycle (`attach` / phase hooks) is best-effort and cannot crash training.
- **channels_last**: Architecture dead-filter checks now support non-contiguous weights.
- **Baseline**: Invalid baseline files emit a warning instead of silently disabling baselines.
- **SARIF**: Deterministic output ordering for cleaner CI diffs.
