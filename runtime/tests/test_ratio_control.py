import unittest

from memserve_runtime import (
    BandwidthBalancePolicy,
    RatioController,
    RuntimeObserver,
    RuntimeSample,
)


class FakeRuntime:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.updates = []

    def get_ratio(self):
        return self.ratio

    def set_ratio(self, value, stream=None):
        self.ratio = value
        self.updates.append((value, stream))


class RatioControlTests(unittest.TestCase):
    def test_policy_raises_ratio_when_link_utilization_exceeds_hbm(self):
        policy = BandwidthBalancePolicy(step=0.1, deadband=0.05)
        sample = RuntimeSample(hbm_utilization=0.2, link_utilization=0.8)

        recommendation = policy.select(sample, 0.5)

        self.assertAlmostEqual(recommendation, 0.6)

    def test_policy_lowers_ratio_when_hbm_utilization_exceeds_c2c(self):
        policy = BandwidthBalancePolicy(step=0.1, deadband=0.05)
        sample = RuntimeSample(hbm_utilization=0.8, link_utilization=0.2)

        recommendation = policy.select(sample, 0.5)

        self.assertAlmostEqual(recommendation, 0.4)

    def test_policy_holds_without_bandwidth_metrics(self):
        policy = BandwidthBalancePolicy(step=0.1, deadband=0.05)
        samples = (
            RuntimeSample(request_latency_ms=12),
            RuntimeSample(hbm_utilization=0.2),
            RuntimeSample(link_utilization=0.8),
        )

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertIsNone(policy.select(sample, 0.5))

    def test_policy_holds_when_bandwidth_difference_is_in_deadband(self):
        policy = BandwidthBalancePolicy(step=0.1, deadband=0.05)
        sample = RuntimeSample(hbm_utilization=0.5, link_utilization=0.52)

        recommendation = policy.select(sample, 0.5)

        self.assertIsNone(recommendation)

    def test_controller_applies_automatic_ratio_at_batch_boundary(self):
        runtime = FakeRuntime()
        controller = RatioController(
            runtime,
            RuntimeObserver(ema_weight=1.0),
            BandwidthBalancePolicy(step=0.1, deadband=0.05),
            update_interval_s=0,
        )
        stream = object()
        controller.observe(RuntimeSample(hbm_utilization=0.2, link_utilization=0.8))

        self.assertEqual(runtime.ratio, 0.5)
        self.assertAlmostEqual(controller.before_batch(stream), 0.6)
        self.assertEqual(runtime.updates[-1][1], stream)

    def test_controller_applies_manual_ratio_on_next_batch(self):
        runtime = FakeRuntime()
        controller = RatioController(
            runtime,
            RuntimeObserver(ema_weight=1.0),
            BandwidthBalancePolicy(step=0.1, deadband=0.05),
            update_interval_s=0,
        )
        controller.observe(RuntimeSample(hbm_utilization=0.2, link_utilization=0.8))

        controller.set_manual_ratio(0.2)

        self.assertAlmostEqual(runtime.ratio, 0.5)
        self.assertEqual(controller.status().pending_ratio, 0.2)
        self.assertAlmostEqual(controller.before_batch(), 0.2)

    def test_controller_resumes_policy_after_manual_ratio_is_cleared(self):
        runtime = FakeRuntime()
        controller = RatioController(
            runtime,
            RuntimeObserver(ema_weight=1.0),
            BandwidthBalancePolicy(step=0.1, deadband=0.05),
            update_interval_s=0,
        )
        controller.observe(RuntimeSample(hbm_utilization=0.2, link_utilization=0.8))
        controller.set_manual_ratio(0.2)
        controller.before_batch()

        controller.clear_manual_ratio()

        self.assertAlmostEqual(controller.before_batch(), 0.3)

    def test_invalid_policy_result_does_not_touch_runtime(self):
        class BadPolicy:
            def select(self, sample, current_ratio):
                return 1.5

        runtime = FakeRuntime()
        controller = RatioController(runtime, RuntimeObserver(), BadPolicy(), 0)
        with self.assertRaises(ValueError):
            controller.before_batch()
        self.assertEqual(runtime.updates, [])


if __name__ == "__main__":
    unittest.main()
