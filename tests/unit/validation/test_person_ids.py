from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import AdapterOutputValidator, OutputValidationCode


def make_output(*person_ids):
    keypoints = tuple(Keypoint(name=name, x=0.5, y=0.5) for name in DUMMY_5_SCHEMA.keypoint_names)
    return AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=tuple(
            PosePrediction(keypoints=keypoints, person_id=person_id) for person_id in person_ids
        ),
    )


def test_accepts_missing_person_ids():
    assert AdapterOutputValidator().validate_person_ids(make_output(None, None)) == []


def test_accepts_unique_person_ids():
    assert AdapterOutputValidator().validate_person_ids(make_output("person-1", "person-2")) == []


def test_rejects_empty_person_id():
    issues = AdapterOutputValidator().validate_person_ids(make_output(""))
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.EMPTY_PERSON_ID
    assert issues[0].path == "poses[0].person_id"


def test_rejects_duplicate_person_ids():
    issues = AdapterOutputValidator().validate_person_ids(make_output("person-1", "person-1"))
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.DUPLICATE_PERSON_ID
    assert issues[0].path == "poses[1].person_id"
    assert "person-1" in issues[0].message


def test_multiple_poses_are_valid():
    output = make_output(None, "person-1", None, "person-2")
    assert AdapterOutputValidator().validate_person_ids(output) == []


def test_zero_poses_is_valid():
    assert AdapterOutputValidator().validate_person_ids(make_output()) == []


def test_accepts_same_person_id_across_inference_outputs():
    validator = AdapterOutputValidator()
    first_image_output = make_output("person-1", "person-2")
    second_image_output = make_output("person-1", "person-3")
    assert validator.validate_person_ids(first_image_output) == []
    assert validator.validate_person_ids(second_image_output) == []


def test_duplicate_failure_does_not_affect_later_inference_output():
    validator = AdapterOutputValidator()
    assert validator.validate_person_ids(make_output("person-1", "person-1"))
    assert validator.validate_person_ids(make_output("person-1")) == []


def test_collects_all_person_id_issues():
    output = make_output("person-1", None, "", "person-1", "person-2", "person-1", "")
    issues = AdapterOutputValidator().validate_person_ids(output)
    assert [(issue.path, issue.code) for issue in issues] == [
        ("poses[2].person_id", OutputValidationCode.EMPTY_PERSON_ID),
        ("poses[3].person_id", OutputValidationCode.DUPLICATE_PERSON_ID),
        ("poses[5].person_id", OutputValidationCode.DUPLICATE_PERSON_ID),
        ("poses[6].person_id", OutputValidationCode.EMPTY_PERSON_ID),
    ]
