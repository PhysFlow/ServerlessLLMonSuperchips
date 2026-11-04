import pytest

from memserve_control_plane import errors

from memserve_control_plane.errors import (
    ControlPlaneError,
    InvalidRequest,
    SchedulableModelNotFound,
    StaleModelPlacementReservation,
    WorkerStateInconsistent,
)


@pytest.mark.parametrize(
    ("error_type", "code", "http_status", "retryable"),
    [
        (SchedulableModelNotFound, "schedulable_model_not_found", 404, False),
        (InvalidRequest, "invalid_request", 400, False),
        (StaleModelPlacementReservation, "stale_model_placement_reservation", 503, True),
        (WorkerStateInconsistent, "worker_state_inconsistent", 503, False),
    ],
)
def test_errors_expose_stable_boundary_metadata(
    error_type: type[ControlPlaneError],
    code: str,
    http_status: int,
    retryable: bool,
) -> None:
    error = error_type("request failed", details={"request_id": "request-1"})

    assert error.code == code
    assert error.http_status == http_status
    assert error.retryable is retryable
    assert error.details == (("request_id", "request-1"),)
    assert str(error) == "request failed"


def test_error_details_are_copied_to_immutable_pairs() -> None:
    details = {"request_id": "request-1"}
    error = ControlPlaneError("failed", details=details)

    details["request_id"] = "request-2"
    assert error.details == (("request_id", "request-1"),)


def test_error_details_require_string_keys_and_values() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        ControlPlaneError("failed", details={"request_id": 12})  # type: ignore[dict-item]


@pytest.mark.parametrize("name,code,status,retryable", [
    ("ControlPlaneError", "control_plane_error", 500, False),
    ("SchedulableModelNotFound", "schedulable_model_not_found", 404, False),
    ("InvalidRequest", "invalid_request", 400, False),
    ("NoFeasibleModelPlacement", "no_feasible_model_placement", 503, False),
    ("StaleModelPlacementReservation", "stale_model_placement_reservation", 503, True),
    ("WorkerUnavailable", "worker_unavailable", 503, True),
    ("WorkerStateInconsistent", "worker_state_inconsistent", 503, False),
    ("WorkerStartupFailed", "worker_startup_failed", 503, False),
    ("ModelDisabled", "model_disabled", 503, False),
    ("WorkerRequestFailed", "worker_request_failed", 502, False),
])
def test_all_error_contracts(name, code, status, retryable):
    error = getattr(errors, name)("failed")
    assert isinstance(error, ControlPlaneError)
    assert (error.code, error.http_status, error.retryable) == (code, status, retryable)
    assert error.details == ()
    assert error.message == "failed"
    assert error.args == ("failed",)


@pytest.mark.parametrize("details", [{1: "value"}, {"key": None}])
def test_error_rejects_nonstring_detail_keys_and_values(details):
    with pytest.raises(TypeError, match="must be strings"):
        ControlPlaneError("failed", details=details)


def test_error_details_have_stable_sorted_order():
    error = ControlPlaneError("failed", details={"request_id": "r", "plan_id": "p"})
    assert error.details == (("plan_id", "p"), ("request_id", "r"))
