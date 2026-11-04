from __future__ import annotations

import logging
import threading
import time
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .runtime_observer import RuntimeObserver, RuntimeSample
from .ratio_policy import RatioPolicy
from .ratio_validation import validate_ratio

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .runtime import Runtime


@dataclass(frozen=True)
class ControllerStatus:
    ratio: float
    manual_ratio: float | None
    pending_ratio: float | None
    last_sample: RuntimeSample
    last_update_ns: int | None


class RatioController:

    def __init__(
        self,
        runtime: Runtime,
        observer: RuntimeObserver,
        policy: RatioPolicy,
        update_interval_s: float = 1.0,
    ):
        if not math.isfinite(update_interval_s) or update_interval_s < 0:
            raise ValueError("update_interval_s must be finite and nonnegative")
        self.runtime = runtime
        self.observer = observer
        self.policy = policy
        self.evaluation_interval_ns = int(update_interval_s * 1e9)
        self._lock = threading.RLock()
        self._manual_ratio: float | None = None
        self._pending_ratio: float | None = None
        self._last_update_ns: int | None = None
        self._last_policy_evaluation_ns: int | None = None
        self._reason = ""

    def observe(self, sample: RuntimeSample) -> None:
        self.observer.observe(sample)

    def set_manual_ratio(self, value: float) -> None:
        with self._lock:
            self._manual_ratio = validate_ratio(value)
            self._pending_ratio = self._manual_ratio

    def clear_manual_ratio(self) -> None:
        with self._lock:
            self._manual_ratio = None
            self._pending_ratio = None
            self._last_policy_evaluation_ns = None

    def before_batch(self, stream=None) -> float:
        with self._lock:
            now_ns = time.monotonic_ns()
            current_ratio = self.runtime.get_ratio()
            candidate_ratio = self._select_ratio_for_batch(now_ns, current_ratio)
            if candidate_ratio is not None:
                self._pending_ratio = validate_ratio(candidate_ratio)
            self._apply_pending_ratio(current_ratio, now_ns, stream)
            return self.runtime.get_ratio()

    def _select_ratio_for_batch(
        self, now_ns: int, current_ratio: float
    ) -> float | None:
        if self._manual_ratio is not None:
            self._reason = "manual"
            return self._manual_ratio
        if (
            self._last_policy_evaluation_ns is not None
            and now_ns - self._last_policy_evaluation_ns < self.evaluation_interval_ns
        ):
            return None
        self._last_policy_evaluation_ns = now_ns
        sample = self.observer.snapshot(now_ns)
        self._reason = f"hbm={_fraction(sample.hbm_utilization)} link={_fraction(sample.link_utilization)}"
        return self.policy.select(sample, current_ratio)

    def _apply_pending_ratio(self, current_ratio: float, now_ns: int, stream) -> None:
        if self._pending_ratio is not None and self._pending_ratio != current_ratio:
            self.runtime.set_ratio(self._pending_ratio, stream=stream)
            logger.info(
                "MemServe asym_ratio %.3f -> %.3f (%s)",
                current_ratio,
                self._pending_ratio,
                self._reason,
            )
            self._last_update_ns = now_ns
        self._pending_ratio = None

    def status(self) -> ControllerStatus:
        with self._lock:
            return ControllerStatus(
                ratio=self.runtime.get_ratio(),
                manual_ratio=self._manual_ratio,
                pending_ratio=self._pending_ratio,
                last_sample=self.observer.snapshot(),
                last_update_ns=self._last_update_ns,
            )


def _fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
