# v1 requirements

## Requirements format and priorities

Priorities are defined as:
- **P0 (must):** required for v1 MVP.
- **P1 (should):** strongly recommended in v1, but can be trimmed if time collapses.
- **P2 (could):** valuable, but safe to postpone.


## Functional requirements

| ID | Requirement | Priority | Estimate (hrs) | Acceptance criteria |
|---|---|---:|---:|---|
| FR-01 | Config-driven evaluation run (single command) | P0 | 3 | Given a config file path, the tool loads it, validates it, runs evaluation, returns exit code 0 on pass and non-zero on fail. |
| FR-02 | Standard “ModelAdapter” interface | P0 | 4 | An adapter protocol/interface exists with explicit methods (e.g., `load()`, `infer(input)`, `metadata()`), and at least one built-in adapter implements it; unit tests confirm contract behavior. |
| FR-03 | Built-in “DummyAdapter” for CI and docs | P0 | 2 | Dummy adapter runs on any OS/CI runner without external model deps; produces deterministic fake keypoints; used in CI smoke eval. |
| FR-04 | Input discovery and iteration | P0 | 3 | Tool accepts an input directory and enumerates candidate files deterministically (stable ordering); adapter receives file paths (or bytes) as defined; tests verify ordering and filtering. |
| FR-05 | Execution plan (warmup + timed runs) | P0 | 4 | Config supports warmup iterations and measured iterations; metrics computed only on measured segment; tests validate timing path and counts. |
| FR-06 | Metric: latency distribution | P0 | 4 | Report includes p50/p95/p99 latency for inference calls; correctness validated with deterministic dummy timings in tests. |
| FR-07 | Metric: stability/error rate | P0 | 3 | Report includes count and rate of inference failures/exceptions; failures are captured with error summaries; tests simulate failures. |
| FR-08 | Metric: output schema validation | P0 | 4 | Output validator checks required fields (e.g., N keypoints, coordinate ranges if defined); failures are reported as “invalid output”; tests cover valid/invalid cases. *(Exact schema is currently unspecified; v1 should default to a minimal schema and allow override.)* |
| FR-09 | Gate evaluation (thresholds) | P0 | 4 | Config defines thresholds (e.g., `p95_latency_ms <= X`, `error_rate <= Y`); tool computes pass/fail per gate and overall; exit code reflects outcome. |
| FR-10 | Report generation (JSON + Markdown) | P0 | 5 | Tool writes a deterministic JSON report and a human-readable Markdown summary to an output directory; reports include config digest and environment metadata. |
| FR-11 | CLI UX suitable for CI | P0 | 3 | CLI supports `--config`, `--output-dir`, `--fail-fast` (optional), `--verbose`; examples added to README; command returns stable exit codes. |
| FR-12 | Adapter selection/registration | P1 | 4 | Config can reference adapter by import path (e.g., `package.module:ClassName`) and pass parameters; errors are clear if import fails; tests cover. |
| FR-13 | Multi-model comparison in one run | P1 | 5 | Config can list multiple model entries; tool produces per-model section reports and a comparison table; gates apply per model. |
| FR-14 | Optional JUnit-style output mapping | P2 | 4 | Tool can emit a JUnit-like XML (or a minimal placeholder) so CI systems can ingest results; may be postponed if JSON/MD is sufficient. (pytest supports JUnit XML for pytest tests; this requirement is about *evaluation runs*, not unit tests.) |


## Non-functional requirements and quality targets

| ID | Requirement | Priority | Estimate (hrs) | Acceptance criteria |
|---|---|---:|---:|---|
| NFR-01 | Cross-platform dev support (Linux/macOS) | P0 | 1 | No OS-specific assumptions in paths/processes; CI includes at least one Linux run; docs state supported OS. |
| NFR-02 | Deterministic evaluation runs (where possible) | P0 | 3 | Given same config + dummy adapter + same inputs, report JSON is bit-for-bit stable (timestamps excluded or normalized); tests enforce determinism. |
| NFR-03 | Clear error handling and diagnosability | P0 | 3 | On config/adapter/input failures, CLI prints actionable messages and returns a distinct non-zero exit code; tests cover main failure classes. |
| NFR-04 | Code quality gates enforced locally and in CI | P0 | 2 | `pre-commit run --all-files` and `pytest` are required checks in CI and documented locally; already present; keep intact while expanding. |
| NFR-05 | Test coverage targets for v1 core | P0 | 2 | Enforce ≥85% line coverage and ≥70% branch coverage on core evaluation modules using coverage tooling; CI fails under threshold. |
| NFR-06 | Minimal runtime dependencies | P0 | 1 | v1 core runs with zero external deps beyond current toolchain; any heavy model runtime deps are optional extras or adapter-level deps.  |
| NFR-07 | Packaging/build sanity | P1 | 2 | CI includes a packaging check build step (e.g., `python -m build`), ensuring `hatchling` backend remains valid.  |
| NFR-08 | Documentation completeness for “public API” | P0 | 4 | README includes: quickstart, config reference, adapter authoring guide, report format; docs match current CLI options; semver policy documented. |
