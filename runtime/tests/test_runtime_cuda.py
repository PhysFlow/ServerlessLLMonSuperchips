import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
class RuntimeCudaTests(unittest.TestCase):
    def test_sglang_wrapper_uses_runtime_ratio_tensor(self):
        if torch.cuda.get_device_capability(0) != (9, 0):
            self.skipTest("SGLang HybridGEMM wrapper requires SM90")
        try:
            from sglang.srt.layers.hybrid_gemm_wrapper import (
                apply_hybrid_gemm,
                set_ratio_handle,
            )
            from sglang.srt.layers.hybrid_gemm_wrapper import entrypoint
        except ImportError:
            self.skipTest("SGLang is not installed")
        from memserve_runtime import Runtime

        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "0"}):
            runtime = Runtime("cuda:0", 0.0)
        captured = []
        fake_extension = ModuleType("hybrid_gemm")

        def fake_hybrid_gemm(x, weight, ratio):
            captured.append(ratio)
            return torch.empty(
                x.shape[0], weight.shape[0], device=x.device, dtype=x.dtype
            )

        fake_extension.hybrid_gemm = fake_hybrid_gemm
        x = torch.empty(1, 256, device="cuda:0", dtype=torch.bfloat16)
        weight = torch.empty(256, 256, dtype=torch.bfloat16, pin_memory=True)
        try:
            set_ratio_handle(runtime.get_ratio_handle())
            with (
                patch.object(
                    entrypoint,
                    "get_global_server_args",
                    return_value=SimpleNamespace(hybrid_gemm_asym_ratio=0.0),
                ),
                patch.dict(sys.modules, {"hybrid_gemm": fake_extension}),
            ):
                apply_hybrid_gemm(x, weight)
                runtime.set_manual_ratio(1.0)
                runtime.before_batch(torch.cuda.current_stream())
                apply_hybrid_gemm(x, weight)
            self.assertEqual(len(captured), 2)
            self.assertIs(captured[0], runtime.get_ratio_handle().tensor)
            self.assertIs(captured[0], captured[1])
            self.assertAlmostEqual(captured[1].item(), 1.0)
        finally:
            set_ratio_handle(None)
            runtime.close()

    def test_handle_keeps_same_tensor_during_manual_and_automatic_updates(self):
        from memserve_runtime import Runtime, RuntimeSample

        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "0"}):
            runtime = Runtime("cuda:0", 0.5, update_interval_s=0)
        try:
            handle = runtime.get_ratio_handle()
            tensor = handle.tensor
            pointer = tensor.data_ptr()
            stream = torch.cuda.Stream(device=runtime.device)

            runtime.set_manual_ratio(0.25)
            self.assertEqual(runtime.get_ratio(), 0.5)
            runtime.before_batch(stream)
            runtime.clear_manual_ratio()
            runtime.observe(RuntimeSample(hbm_utilization=0.2, link_utilization=0.8))
            runtime.before_batch(stream)
            stream.synchronize()

            self.assertIs(runtime.get_ratio_handle(), handle)
            self.assertIs(handle.tensor, tensor)
            self.assertEqual(handle.tensor.data_ptr(), pointer)
            self.assertAlmostEqual(runtime.get_ratio(), 0.3)
            self.assertAlmostEqual(tensor.item(), 0.3, places=5)
        finally:
            runtime.close()
        with self.assertRaises(RuntimeError):
            _ = handle.tensor

    def test_hybridgemm_graph_replay_reads_runtime_updated_ratio(self):
        if torch.cuda.get_device_capability(0) != (9, 0):
            self.skipTest("SM90 HybridGEMM kernel required")
        try:
            from hybrid_gemm import hybrid_gemm
        except ImportError:
            self.skipTest("HybridGEMM extension is not installed")
        from memserve_runtime import Runtime

        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "0"}):
            runtime = Runtime("cuda:0", 0.0)
        try:
            x = torch.randn(2048, 256, device="cuda:0", dtype=torch.bfloat16)
            weight = torch.randn(2048, 256, dtype=torch.bfloat16, pin_memory=True)
            handle = runtime.get_ratio_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = hybrid_gemm(x, weight, handle.tensor)

            expected = x.float() @ weight.cuda().float().T
            pointer = handle.tensor.data_ptr()
            for ratio in (0.0, 0.5, 1.0):
                runtime.set_manual_ratio(ratio)
                runtime.before_batch(torch.cuda.current_stream())
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(handle.tensor.data_ptr(), pointer)
                torch.testing.assert_close(
                    actual.float(), expected, rtol=1e-2, atol=1e-2
                )
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
