# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [0.3.0] - Release

### Added
- **Rule Packs**: Introduced 5 specialized validator packs with ~15 new rules:
  - `TA1xx`: Stability (NaNs, Gradient Explosion, Dead Units).
  - `TA2xx`: Hardware (Split Brain, Tensor Core Alignment).
  - `TA3xx`: Data Integrity (Device Mismatch, Flat Inputs).
  - `TA4xx`: Architecture (Redundant Bias, Zombie Layers).
  - `TA4xx`: Optimization (AdamW usage, Weight Decay targets).
- **CLI**: New `torch-audit` command line tool for static analysis of models.
- **Reporting**: Added SARIF output (`--format sarif`) for GitHub Code Scanning integration.
- **Reporting**: Added JSON output (`--format json`) for CI pipelines.
- **Architecture**: Added `batch` and `optimizer` support to `AuditContext`.

### Changed
- **Core**: Refactored `AuditResult` to track `new_findings` vs `all_findings` (Baseline gating).
- **Core**: Fingerprints are now strictly versioned (`v1:...`) for stable baselines.
- **Config**: Suppression regexes now fail fast on invalid syntax instead of silently ignoring rules.
