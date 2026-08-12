"""Public Python API for near-lossless context compression.

The source remains alongside the research benchmarks so the original experiment
commands keep working. Packaging maps this directory to ``nearlossless_context``.
"""

from __future__ import annotations

from . import adaptive as _adaptive

from .compress_adaptive import (
    prefill_auto,
    prefill_posthoc_adaptive,
    prefill_stream_adaptive,
)
from .decode_utils import greedy_generate

AdaptivePolicy = _adaptive.AdaptivePolicy
policy_for = _adaptive.policy_for

__all__ = [
    "AdaptivePolicy",
    "greedy_generate",
    "policy_for",
    "prefill_auto",
    "prefill_posthoc_adaptive",
    "prefill_stream_adaptive",
]

__version__ = "0.1.1"
