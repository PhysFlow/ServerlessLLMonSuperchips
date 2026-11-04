from __future__ import annotations

from collections.abc import Mapping


class ControlPlaneError(Exception):
    code = "control_plane_error"
    http_status = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = self._freeze_details(details)

    @staticmethod
    def _freeze_details(details: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
        if details is None:
            return ()
        entries = tuple(details.items())
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in entries):
            raise TypeError("error detail keys and values must be strings")
        return tuple(sorted(entries))


class SchedulableModelNotFound(ControlPlaneError):
    code = "schedulable_model_not_found"
    http_status = 404


class InvalidRequest(ControlPlaneError):
    code = "invalid_request"
    http_status = 400


class NoFeasibleModelPlacement(ControlPlaneError):
    code = "no_feasible_model_placement"
    http_status = 503


class StaleModelPlacementReservation(ControlPlaneError):
    code = "stale_model_placement_reservation"
    http_status = 503
    retryable = True


class WorkerUnavailable(ControlPlaneError):
    code = "worker_unavailable"
    http_status = 503
    retryable = True


class WorkerStateInconsistent(ControlPlaneError):
    code = "worker_state_inconsistent"
    http_status = 503


class WorkerStartupFailed(ControlPlaneError):
    code = "worker_startup_failed"
    http_status = 503


class ModelDisabled(ControlPlaneError):
    code = "model_disabled"
    http_status = 503
    retryable = False


class WorkerRequestFailed(ControlPlaneError):
    code = "worker_request_failed"
    http_status = 502
    retryable = False
