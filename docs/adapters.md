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

The only built-in adapter in the v0.7 development milestone is `dummy`.

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
| `schema` | `KeypointSchema` | Declared keypoint names and their exact order for every pose. |
| `poses` | `tuple[PosePrediction, ...]` | Zero or more predicted people for the image. |
| `metadata` | `Mapping[str, Any]` | Adapter-specific metadata for debugging or reporting. |

Each `PosePrediction` contains:

- `keypoints`: `tuple[Keypoint, ...]`
- `confidence`: overall pose confidence
- `person_id`: optional non-empty identifier, unique within this image's output;
  the same ID can appear in outputs for other images

Each `Keypoint` contains:

- `name`: semantic keypoint name such as `nose`
- `x`: x-coordinate or `None`
- `y`: y-coordinate or `None`
- `confidence`: optional confidence for that keypoint
- `visible`: optional visibility flag

The [normalized output contract](output-contract.md) defines the rules enforced
in v0.7:

- Coordinates are finite and normalized to `[0, 1]` relative to the original
  input image, with a top-left origin, positive x right, and positive y down.
- Missing keypoints retain their declared name and position with `x=None` and
  `y=None`. They cannot have `visible=True`.
- Every pose contains exactly `schema.keypoint_names` in deterministic order.
- Pose and keypoint confidence are optional; supplied values are finite and
  within `[0, 1]`.
- Zero or more poses are valid. Non-`None` person IDs are non-empty and unique
  within one output, and may repeat across images.

Adapters must convert native runtime output into this contract, including
undoing crop, resize, or padding transforms. The PDG validator checks supplied
values and structure; it does not perform model-specific normalization.

`DummyAdapter` declares `DUMMY_5_SCHEMA`. `COCO_17_SCHEMA` and custom
`KeypointSchema` instances are also available. A declared schema defines names
and order; it does not require every adapter to share one universal taxonomy.
v0.8 reference comparison will only compare outputs whose schemas match.

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

The adapter interface and normalized output contract do not yet define:

- a plugin or import-path based adapter loading system
- a canonical body-keypoint taxonomy shared by all runtimes
- a required image decoding or tensor-preprocessing API
- batching, streaming, or async inference contracts
- report schema requirements for adapter metadata
- any production pose-model integration

These remain future work beyond the v0.7 normalized output contract.

## Future Adapter Examples

Likely future adapters include:

- an Ultralytics YOLO pose adapter
- a MediaPipe pose adapter
- a Torch-based custom checkpoint adapter
- a remote inference service adapter

Those are examples only. They are not implemented in the v0.7 milestone, and this
document should not be read as a compatibility promise for any specific
runtime.
