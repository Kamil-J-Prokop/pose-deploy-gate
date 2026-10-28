# Configuration examples

Run examples from the repository root. Relative paths are interpreted from the
working directory, not from the YAML file's directory. All examples currently
use `DummyAdapter`: it returns fixed synthetic poses without decoding files.
Changing its confidence parameters does not create geometric pose differences.
No real model adapter, ground-truth accuracy evaluation, temporal tracking, or
3D comparison is demonstrated by these configs.

## Runnable smoke examples

These use repository fixture paths and need no input-path edits:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.comparison-smoke.yaml -v --list-inputs
uv run python -m pose_deploy_gate --config docs/examples/config.comparison-strict.yaml -v
```

The first configured model is the reference; candidates run sequentially after
the single reference run. Both examples compare two separate dummy instances
over three JPEG fixture paths. The default/common example and explicit strict
example both compare the same `dummy_5` schema, so they have no exclusions and
zero keypoint deviation. Different-schema intersections and reversed person
ordering are exercised by synthetic adapters in the regression suite:

```bash
uv run pytest -q tests/integration/test_multi_model_workflows.py
```

## Editable templates

| File | Purpose | Input setup |
| --- | --- | --- |
| [config.comparison.yaml](config.comparison.yaml) | One reference and two candidates, explicit common comparison settings. | Set `data.input_dir` to a directory containing JPEG files. |
| [config.multiple-models.yaml](config.multiple-models.yaml) | Three model entries with shared defaults. | Select your input directory and file pattern. |
| [config.minimal.yaml](config.minimal.yaml) | One reference, no candidates, default runner settings. | Select your input directory; `*` matches all files. |
| [config.full.yaml](config.full.yaml) | Every current config field, including reserved metadata. | Select a directory containing JPEG files. |
| [config.runner.yaml](config.runner.yaml) | Explicit warmup and failure policy. | Select your input directory and file pattern. |
| [config.jpg-only.yaml](config.jpg-only.yaml) | Non-recursive JPEG discovery. | Replace `./images` with your input directory. |
| [config.recursive.yaml](config.recursive.yaml) | Recursive JPEG discovery. | Replace `./images` with your input directory. |

`output.dir`, `gates.enabled`, `run.seed`, and `run.device` do not create reports,
evaluate gates, seed adapters, or choose devices in the current session. The CLI
returns `0` after completion and `2` for fatal evaluation errors; no deployment
PASS/FAIL decision is implemented. See [config](../config.md),
[comparison semantics](../comparison.md), and [evaluation output](../evaluation.md).
