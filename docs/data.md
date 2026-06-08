# Data Sources

Version `0.4.0` introduces deterministic file iteration.

## What It Does

- validates `input_dir`
- discovers files matching `file_pattern`
- optionally recurses into nested directories
- sorts matches by normalized relative POSIX path
- emits `ImageInput(image_id, path)` values

## What It Does Not Do

- decode images
- validate file contents
- load annotations
- run adapters
- batch data

## How Discovery Works

`discover_files()` scans the configured `data.input_dir`, filters results down
to files only, and returns them in deterministic order. Deterministic ordering
matters because filesystem iteration order can vary by operating system,
filesystem, and environment. By normalizing each file to its relative POSIX
path before sorting, the project gets stable input ordering across machines.

`FileDataSource` builds on top of this discovery step and turns each file path
into an `ImageInput`:

- `image_id` is the relative path without the file suffix
- `path` is the actual discovered `Path`

For example, `images/session_a/frame_001.jpg` becomes:

- `image_id="session_a/frame_001"`
- `path=Path("images/session_a/frame_001.jpg")`

## Example Config

[config.recursive.yaml](examples/config.recursive.yaml)

```yaml
version: 1

data:
  input_dir: "./images"
  file_pattern: "*.jpg"
  recursive: true

adapter:
  type: "dummy"
```

## CLI Behavior

When you validate a config through the CLI, PoseDeployGate now reports how many
input files were discovered:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml
```

Add `--list-inputs` to print the deterministic relative file list:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml --list-inputs
```

## Example Files

- Recursive JPEG example: [config.recursive.yaml](examples/config.recursive.yaml)
- JPEG-only example: [config.jpg-only.yaml](examples/config.jpg-only.yaml)
- Minimal example: [config.minimal.yaml](examples/config.minimal.yaml)
