"""Exceptions for adapter output validation."""

from collections.abc import Iterable

from pose_deploy_gate.adapters.exceptions import AdapterError
from pose_deploy_gate.validation.issues import OutputValidationIssue


class AdapterOutputValidationError(AdapterError):
    """Raised when adapter output contains one or more validation issues."""

    def __init__(self, issues: Iterable[OutputValidationIssue]) -> None:
        self.issues = tuple(issues)
        details = "\n".join(
            f"- {issue.path}: {issue.code}: {issue.message}" for issue in self.issues
        )
        message = f"Adapter output validation failed with {len(self.issues)} issues:"
        if details:
            message += f"\n{details}"
        super().__init__(message)
