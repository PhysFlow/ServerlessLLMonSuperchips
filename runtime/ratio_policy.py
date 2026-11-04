from __future__ import annotations

from typing import Protocol

from .runtime_observer import RuntimeSample


class RatioPolicy(Protocol):
    def select(self, sample: RuntimeSample, current_ratio: float) -> float | None:
        ...


class BandwidthBalancePolicy:
    def __init__(
        self,
        step: float = 0.05,
        deadband: float = 0.10,
        target_latency_ms: float | None = None,
    ):
        self.step = step
        self.deadband = deadband
        self.target_latency_ms = target_latency_ms

    def select(self, sample: RuntimeSample, current_ratio: float) -> float | None:
        adjustment_signal = self._adjustment_signal(sample)
        if adjustment_signal is None or abs(adjustment_signal) <= self.deadband:
            return None
        effective_step = self._effective_step(sample)
        candidate_ratio = (
            min(1.0, current_ratio + effective_step)
            if adjustment_signal > 0
            else max(0.0, current_ratio - effective_step)
        )
        return candidate_ratio if candidate_ratio != current_ratio else None

    def _adjustment_signal(self, sample: RuntimeSample) -> float | None:
        if sample.hbm_utilization is None or sample.link_utilization is None:
            return None
        return sample.link_utilization - sample.hbm_utilization

    def _effective_step(self, sample: RuntimeSample) -> float:
        if (
            self.target_latency_ms is None
            or sample.request_latency_ms is None
            or sample.request_latency_ms <= self.target_latency_ms
        ):
            return self.step
        return min(1.0, 2 * self.step)
