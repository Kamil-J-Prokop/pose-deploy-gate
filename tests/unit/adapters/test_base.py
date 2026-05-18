from pathlib import Path

import pytest

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput


def test_pose_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        PoseAdapter()


@pytest.mark.parametrize(
    ("adapter_cls", "missing_member"),
    [
        (
            type(
                "MissingNameAdapter",
                (PoseAdapter,),
                {
                    "predict": lambda self, image: AdapterOutput(poses=()),
                },
            ),
            "name",
        ),
        (
            type(
                "MissingPredictAdapter",
                (PoseAdapter,),
                {
                    "name": property(lambda self: "missing-predict"),
                },
            ),
            "predict",
        ),
    ],
)
def test_concrete_adapter_must_implement_name_and_predict(adapter_cls, missing_member):
    with pytest.raises(TypeError, match=missing_member):
        adapter_cls()


def test_concrete_adapter_can_satisfy_the_interface():
    class ExampleAdapter(PoseAdapter):
        @property
        def name(self) -> str:
            # Replace this with your real adapter identifier when you implement
            # a production adapter class.
            return "example"

        def predict(self, image: ImageInput) -> AdapterOutput:
            # Replace this placeholder with model inference that maps into the
            # shared AdapterOutput shape expected by the rest of the app.
            return AdapterOutput(poses=(), metadata={"image_id": image.image_id})

    adapter = ExampleAdapter()
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))

    output = adapter.predict(image)

    assert adapter.name == "example"
    assert output == AdapterOutput(poses=(), metadata={"image_id": "image-001"})
