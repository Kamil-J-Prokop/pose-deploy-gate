# Data Sources

`FileDataSource`, introduced in v0.4, supplies deterministic file iteration to
every model in a sequential evaluation.

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

Input paths are resolved from the process working directory. Because image IDs
omit suffixes, `frame.jpg` and `frame.png` in the same directory produce the same
ID. Discovery itself can return both; `EvaluationSession` rejects their duplicate
ID. Use a filter such as `*.jpg` or inputs with distinct suffix-free paths.

The session shares one data-source instance across models, but each runner
rediscovers files before inference. Keep the input set and file contents stable.
Different candidate/reference ID sets raise a consistency error, including when
the missing ID belonged to a failed prediction. No content hashes are computed.

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

models:
  - id: dummy
    adapter:
      type: "dummy"
```

## CLI Behavior

When you run a config through the CLI, PoseDeployGate reports how many input
attempts each model made after the evaluation completes:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml
```

Add `--list-inputs` to print the deterministic relative file list:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml --list-inputs
```

The list is printed once using reference prediction paths, after all candidate
summaries. The [two-model smoke example](examples/config.comparison-smoke.yaml)
uses repository fixtures and is runnable from the repository root.

## Example Files

- Recursive JPEG example: [config.recursive.yaml](examples/config.recursive.yaml)
- JPEG-only example: [config.jpg-only.yaml](examples/config.jpg-only.yaml)
- Minimal example: [config.minimal.yaml](examples/config.minimal.yaml)
