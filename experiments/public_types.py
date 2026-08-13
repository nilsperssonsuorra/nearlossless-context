"""Public typing contracts for prefill metadata.

These definitions describe the dictionaries returned by :func:`prefill_auto`.
They do not change the runtime tuple-and-dictionary API.
"""

from __future__ import annotations

from typing import TypedDict


class PolicyInfo(TypedDict):
    """Resolved adaptive cache policy."""

    R: int
    budget: int
    stream_budget: int
    mode: str
    note: str
    family: str


class _CompressionStatsRequired(TypedDict):
    stream_budget: int
    final_budget: int
    n_compress: int


class CompressionStats(_CompressionStatsRequired, total=False):
    """Streaming statistics shared across discovery implementations."""

    path: str
    peak_cache: int
    final_cache: int
    peak_cache_tokens: int
    final_cache_tokens: int
    final_kv_mb: float
    mode: str
    n_entities_hat: int
    auto_raise_budget: bool
    expand_radius: int
    last_n_novelty_kept: int
    last_n_attn_pins: int
    last_novelty_retention_proxy: float
    floor_ratio: float
    max_capsules: int
    sticky: bool
    n_sticky_pins: int
    hold_factor: float
    hold_budget: int


class _PrefillInfoRequired(TypedDict):
    path: str
    L: int
    policy: PolicyInfo | None
    model_id: str | None


class PrefillInfo(_PrefillInfoRequired, total=False):
    """Metadata returned as the third item from :func:`prefill_auto`."""

    discovery: str
    n_entities_hat: int
    stats: CompressionStats
    keep_count: int
    cache_tokens: int
    logical_kv_mb_int8: float | None
    use_int8: bool
