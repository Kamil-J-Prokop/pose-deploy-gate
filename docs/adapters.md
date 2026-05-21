# Adapter Reference

Adapters are the boundary between PoseDeployGate and a pose-estimation
runtime. The rest of the project should be able to work with normalized inputs
and outputs without knowing whether predictions came from a local model, a
remote service, or a test double.

## What Adapters Are

An adapter is a small object that implements the shared `PoseAdapter`
interface:

- it exposes a stable `name`
- it accepts one normalized `ImageInput`
- it returns one normalized `AdapterOutput`

Today that interface lives in `src/pose_deploy_gate/adapters/base.py`, and the
shared data structures live in `src/pose_deploy_gate/adapters/types.py`.

## Why They Exist

Adapters keep model-specific concerns isolated from the rest of the evaluation
pipeline.

That separation matters because PoseDeployGate needs to compare different pose
runtimes using one consistent contract. A runner, metrics layer, or report
writer should not need custom logic for each model backend.

Adapters also make the project easier to test. We can validate CLI wiring,
config loading, and output normalization without requiring heavyweight ML
dependencies in CI.

## Supported Adapter

The only built-in adapter in `v0.3.0` is `dummy`.

`DummyAdapter` is a deterministic fake adapter used for:

- smoke tests
- contract tests
- local CLI validation
- future pipeline wiring before a real model runtime is added

It always returns the same single-pose prediction shape, with configurable
`keypoint_confidence` and `pose_confidence` values.

## Input And Output Contract

Every adapter must implement:

```python
class PoseAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict(self, image: ImageInput) -> AdapterOutput: ...
```

### Input

`ImageInput` is the normalized per-image input passed to adapters.

| Field | Type | Meaning |
| --- | --- | --- |
| `image_id` | `str` | Stable identifier for the image within a run. |
| `path` | `pathlib.Path` | Filesystem path to the input image. |

Current adapters receive a path reference, not already-decoded image bytes or
tensors.

### Output

`AdapterOutput` is the normalized result for one image.

| Field | Type | Meaning |
| --- | --- | --- |
| `poses` | `tuple[PosePrediction, ...]` | Zero or more predicted people for the image. |
| `metadata` | `Mapping[str, Any]` | Adapter-specific metadata for debugging or reporting. |

Each `PosePrediction` contains:

- `keypoints`: `tuple[Keypoint, ...]`
- `confidence`: overall pose confidence
- `person_id`: optional stable identifier for the predicted person

Each `Keypoint` contains:

- `name`: semantic keypoint name such as `nose`
- `x`: x-coordinate or `None`
- `y`: y-coordinate or `None`
- `confidence`: optional confidence for that keypoint
- `visible`: optional visibility flag

The current dummy adapter returns normalized coordinates in the `[0.0, 1.0]`
range, but the interface does not yet define a project-wide coordinate system
beyond the field names above. That stricter schema is intentionally deferred.

## Example Config

Minimal config using the built-in dummy adapter:

```yaml
version: 1

data:
  input_dir: "."

adapter:
  type: "dummy"
```

Config with explicit dummy parameters:

```yaml
version: 1

data:
  input_dir: "./data"

adapter:
  type: "dummy"
  params:
    keypoint_confidence: 0.9
    pose_confidence: 0.8
```

See also:

- [docs/examples/config.minimal.yaml](examples/config.minimal.yaml)
- [tests/fixtures/config/dummy_with_params.yaml](../tests/fixtures/config/dummy_with_params.yaml)

## Intentionally Out Of Scope

This milestone adds the adapter interface and one deterministic built-in
adapter, but it does not yet define:

- a plugin or import-path based adapter loading system
- a canonical body-keypoint taxonomy shared by all runtimes
- a required image decoding or tensor-preprocessing API
- batching, streaming, or async inference contracts
- report schema requirements for adapter metadata
- any production pose-model integration

Those pieces are easier to finalize once the runner, metrics, and report
pipeline exist.

## Future Adapter Examples

Likely future adapters include:

- an Ultralytics YOLO pose adapter
- a MediaPipe pose adapter
- a Torch-based custom checkpoint adapter
- a remote inference service adapter

Those are examples only. They are not implemented in `v0.3.0`, and this
document should not be read as a compatibility promise for any specific
runtime.
