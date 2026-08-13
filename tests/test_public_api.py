from __future__ import annotations

import unittest

import nearlossless_context as nlc
import torch
from nearlossless_context.example import _resolve_device, build_parser
from nearlossless_context.snapkv import compress_keep_indices


class _Layer:
    def __init__(self, *, sliding: bool = False) -> None:
        values = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
        self.keys = values.clone()
        self.values = values.clone()
        self.is_sliding = sliding


class _Cache:
    def __init__(self) -> None:
        self.layers = [_Layer(), _Layer(sliding=True)]


class PublicApiTests(unittest.TestCase):
    def test_public_api_is_importable(self) -> None:
        self.assertEqual(nlc.__version__, "0.1.1")
        self.assertTrue(callable(nlc.prefill_auto))
        self.assertTrue(callable(nlc.greedy_generate))

    def test_default_policy_is_bounded(self) -> None:
        policy = nlc.policy_for(n_entities=1, L=4096, prefer_stream=True)

        self.assertIsInstance(policy, nlc.AdaptivePolicy)
        self.assertGreater(policy.stream_budget, 0)
        self.assertLess(policy.stream_budget, 4096)
        self.assertGreaterEqual(policy.R, 0)

    def test_cache_selection_preserves_sliding_layers(self) -> None:
        cache = _Cache()

        compress_keep_indices(cache, [1, 3])

        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [1.0, 3.0])
        self.assertEqual(cache.layers[1].keys.shape[-2], 5)

    def test_example_defaults_are_cpu_safe(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.device, "auto")
        self.assertEqual(args.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(_resolve_device("auto", cuda_available=False), "cpu")

    def test_example_rejects_unavailable_cuda(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            _resolve_device("cuda", cuda_available=False)
