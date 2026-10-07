from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import AdapterOutputValidator, OutputValidationCode

SCHEMA = KeypointSchema(
    name="custom_3",
    keypoint_names=("nose", "left_shoulder", "right_shoulder"),
)


def make_pose(names):
    return PosePrediction(
        keypoints=tuple(Keypoint(name=name, x=0.5, y=0.5) for name in names),
    )


def assert_schema_mismatch(names):
    output = AdapterOutput(schema=SCHEMA, poses=(make_pose(names),))
    issues = AdapterOutputValidator().validate_schema(output)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.KEYPOINT_SCHEMA_MISMATCH
    assert issues[0].path == "poses[0].keypoints"
    assert repr(SCHEMA.keypoint_names) in issues[0].message
    assert repr(names) in issues[0].message


def test_accepts_keypoints_matching_schema():
    output = AdapterOutput(schema=SCHEMA, poses=(make_pose(SCHEMA.keypoint_names),))
    assert AdapterOutputValidator().validate_schema(output) == []


def test_rejects_missing_schema_keypoint():
    assert_schema_mismatch(("nose", "left_shoulder"))


def test_rejects_extra_keypoint():
    assert_schema_mismatch((*SCHEMA.keypoint_names, "left_wrist"))


def test_rejects_wrong_keypoint_name():
    assert_schema_mismatch(("nose", "left_shoulder", "left_wrist"))


def test_rejects_duplicate_keypoint():
    assert_schema_mismatch(("nose", "left_shoulder", "left_shoulder"))


def test_rejects_wrong_keypoint_order():
    assert_schema_mismatch(("nose", "right_shoulder", "left_shoulder"))


def test_zero_poses_is_valid():
    output = AdapterOutput(schema=SCHEMA, poses=())
    assert AdapterOutputValidator().validate_schema(output) == []


def test_rejects_pose_with_zero_keypoints():
    assert_schema_mismatch(())


def test_collects_schema_mismatches_across_all_poses():
    output = AdapterOutput(
        schema=SCHEMA,
        poses=(
            make_pose(SCHEMA.keypoint_names),
            make_pose(("nose",)),
            make_pose(SCHEMA.keypoint_names),
            make_pose(("nose", "right_shoulder", "left_shoulder")),
        ),
    )
    issues = AdapterOutputValidator().validate_schema(output)
    assert [issue.path for issue in issues] == [
        "poses[1].keypoints",
        "poses[3].keypoints",
    ]
    assert all(issue.code == OutputValidationCode.KEYPOINT_SCHEMA_MISMATCH for issue in issues)
