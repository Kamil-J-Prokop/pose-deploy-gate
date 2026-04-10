# Config Reference

PoseDeployGate uses a YAML config file to describe an evaluation run: where
input data comes from, which adapter to use, where outputs should be written,
and whether deployment gates are enabled.

## Loading

Configs are loaded with `load_config(config_path)`.

- Supported file extensions are `.yaml` and `.yml`.
- Files must contain a top-level YAML mapping.
- Unknown keys are rejected at every schema level.
- Path values are parsed as `pathlib.Path` values.
- `data.input_dir` must exist and must be a directory when the config is loaded.

## CLI Usage

Validate a config file from the command line:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml
```

The command loads the YAML file, applies schema defaults, runs config
validation, prints the resolved run settings, and exits with `0` when the
config is valid. Config loading failures print an `ERROR:` message and exit
with a non-zero code.

## Minimal Example

Start from the minimal example when you only need the default run, output, and
gate settings:

[config.minimal.yaml](examples/config.minimal.yaml)

```yaml
version: 1

data:
  input_dir: "."

adapter:
  type: "dummy"
```

## Full Example

The full example sets every currently supported section and field:

[config.full.yaml](examples/config.full.yaml)

## Schema

### `version`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `version` | Yes | literal `1` | none | Config schema version. Only `1` is currently supported. |

### `run`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `run.name` | No | string | `default-run` | Must not be empty or whitespace. |
| `run.seed` | No | integer or null | `null` | When set, must be non-negative. |
| `run.device` | No | `cpu`, `cuda`, or `auto` | `cpu` | Selects the runtime device preference. |

### `data`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `data.input_dir` | Yes | path | none | Must exist and must be a directory. |
| `data.file_pattern` | No | string | `*` | Must not be empty or whitespace. |
| `data.recursive` | No | boolean | `false` | Enables recursive input discovery. |

### `adapter`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `adapter.type` | Yes | literal `dummy` | none | Only the dummy adapter is currently supported. |
| `adapter.params` | No | mapping | `{}` | Adapter-specific parameters. |

### `output`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `output.dir` | No | path | `./artifacts` | Output directory path. It is parsed but not required to exist during config loading. |

### `gates`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `gates.enabled` | No | boolean | `false` | Enables deployment gate evaluation. |

## Validation Failures

Config loading raises config-specific exceptions:

- `ConfigFileNotFoundError` when the config file path does not exist.
- `ConfigParseError` when YAML parsing fails, the file is empty, or the top
  level is not a mapping.
- `ConfigValidationError` when the file extension, schema, unknown keys, or
  application validation rules fail.

Common invalid configs include missing required fields, invalid `run.device`
values, unknown keys, empty files, malformed YAML, missing `data.input_dir`,
empty `run.name`, empty `data.file_pattern`, and negative `run.seed`.

## Example Files

- Minimal example: [config.minimal.yaml](examples/config.minimal.yaml)
- Full example: [config.full.yaml](examples/config.full.yaml)
