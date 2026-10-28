| Target version | Key deliverable | Status |
|---|---|---|
| 0.1.0 | Repository structure + working CI gates | Done - 14.03.2026 | 
| 0.2.0 | Config schema + loader + validation | Done - 10.04.2026 |
| 0.3.0 | Adapter interface + DummyAdapter | Done - 21.05.2026 |
| 0.4.0 | DataSource + deterministic file iteration | Done - 09.06.2026 |
| 0.5.0 | Runner with warmup + timing capture | Done - 11.09.2026 |
| 0.6.0 | MetricsEngine (latency + error rate) | Done - 23.09.2026 |
| 0.7.0 | Output validator + schema defaults | Done - 07.10.2026 |
| 0.8.0 | Sequential multi-model reference comparison, common/strict schemas, multi-person matching, quality/coverage metrics, and CLI summaries | Implemented in development; release preparation pending |
| 0.9.0 | GateEvaluator + ReportWriter (JSON + MD), consuming EvaluationResult | Planned |
| 0.10.0 | CI upgrades: coverage + build check + upload artifacts + job summary | Planned |
| 1.0.0 | MVP release - Docs completion + release criteria met + tag v1.0.0 | Planned |

The package version remains `0.7.0` until release preparation. Comparison
semantics are documented in [comparison.md](comparison.md); the roadmap does
not imply that gate evaluation or report artifacts already exist.
