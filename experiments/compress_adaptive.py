"""
End-to-end adaptive compression: prefill → score → estimate entities → policy → keep.

Posthoc path only for entity estimate (needs full past + question window scores).
Streaming uses L-based policy unless n_entities is provided.
"""

from __future__ import annotations

from typing import Any

import torch

from adaptive import AdaptivePolicy, policy_for, policy_from_scores
from scorer_valley import (
    aggregate_prefix_vote,
    compress_with_seed_valley,
    obs_attentions,
    prefill_streaming_valley,
)
from snapkv import cache_seq_len, prefill_chunked


@torch.inference_mode()
def score_prefix_from_past(
    model,
    input_ids: torch.Tensor,
    past,
    *,
    window_size: int = 128,
    score_layers: int = 8,
) -> tuple[torch.Tensor, int, int]:
    """Return (score[prefix], prefix_len, window_size)."""
    attns, window_size, prefix_len = obs_attentions(
        model, input_ids, past, window_size
    )
    if attns is None or all(a is None for a in attns):
        # flat dummy scores
        s = cache_seq_len(past)
        return torch.zeros(max(s, 1), device=input_ids.device), max(s - 1, 0), window_size
    h_kv = past.layers[0].keys.shape[1]
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    return score, prefix_len, window_size


@torch.inference_mode()
def prefill_posthoc_adaptive(
    model,
    input_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    multi_hop: bool = False,
    n_entities: int | None = None,
    min_sep: int = 200,
    score_layers: int = 8,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Chunked full prefill → obs attention → estimate n_entities (optional) →
    adaptive R/budget → seed_valley compress.

    Returns (past, last_logits, info).
    """
    seq_len = int(input_ids.shape[-1])
    past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)

    score, prefix_len, window_size = score_prefix_from_past(
        model,
        input_ids,
        past,
        window_size=window_size,
        score_layers=score_layers,
    )

    if n_entities is not None:
        pol = policy_for(
            n_entities=n_entities,
            L=seq_len,
            multi_hop=multi_hop,
            prefer_stream=False,
        )
        n_hat = n_entities
    else:
        pol, n_hat = policy_from_scores(
            score,
            L=seq_len,
            prefix_len=prefix_len,
            multi_hop=multi_hop,
            prefer_stream=False,
            min_sep=min_sep,
        )

    # Re-compress with policy (re-score inside compress_with_seed_valley)
    past, keep = compress_with_seed_valley(
        model,
        input_ids,
        past,
        budget=pol.budget,
        window_size=window_size,
        sinks=sinks,
        expand_radius=pol.R,
        score_layers=score_layers,
        mode=pol.mode,
    )
    info = {
        "path": "posthoc_adaptive",
        "n_entities_hat": n_hat,
        "policy": pol.__dict__,
        "keep_count": len(keep),
        "cache_tokens": cache_seq_len(past),
        "L": seq_len,
    }
    return past, logits, info


@torch.inference_mode()
def prefill_stream_adaptive(
    model,
    input_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    multi_hop: bool = False,
    n_entities: int | None = None,
    score_layers: int = 8,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Streaming compress with adaptive stream_budget.

    Without n_entities, uses L-only schedule (n_entities=1) then optional
    final tighten is not applied (stream keeps stream_budget).
    Prefer passing n_entities when known (e.g. multi-doc).
    """
    seq_len = int(input_ids.shape[-1])
    n_ent = n_entities if n_entities is not None else 1
    pol = policy_for(
        n_entities=n_ent,
        L=seq_len,
        multi_hop=multi_hop,
        prefer_stream=True,
    )
    past, logits, stats = prefill_streaming_valley(
        model,
        input_ids,
        stream_budget=pol.stream_budget,
        final_budget=pol.stream_budget,
        chunk_size=chunk_size,
        window_size=window_size,
        sinks=sinks,
        expand_radius=pol.R,
        score_layers=score_layers,
        mode=pol.mode,
    )
    info = {
        "path": "stream_adaptive",
        "n_entities_hat": n_ent,
        "policy": pol.__dict__,
        "stats": stats,
        "L": seq_len,
    }
    return past, logits, info


@torch.inference_mode()
def prefill_auto(
    model,
    input_ids: torch.Tensor,
    *,
    mode: str = "stream",
    n_entities: int | None = None,
    multi_hop: bool = False,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
) -> tuple[Any, torch.Tensor, dict]:
    """
    One-call entrypoint for “fit more context under 24GB”.

    mode:
      "stream"  — online compress (low peak cache; default for long L)
      "posthoc" — full prefill then compress (auto n̂ if n_entities is None)
      "full"    — chunked full KV (no compress)

    Recommended:
      long single-doc retrieval → mode="stream", n_entities=1 (or omit)
      multi-secret / multi-doc → mode="stream", n_entities=3 (or more)
      best quality post-score → mode="posthoc"
    """
    if mode == "full":
        past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
        info = {
            "path": "full_chunked",
            "L": int(input_ids.shape[-1]),
            "cache_tokens": cache_seq_len(past),
            "policy": None,
        }
        return past, logits, info
    if mode == "posthoc":
        return prefill_posthoc_adaptive(
            model,
            input_ids,
            chunk_size=chunk_size,
            window_size=window_size,
            sinks=sinks,
            multi_hop=multi_hop,
            n_entities=n_entities,
        )
    if mode == "stream":
        return prefill_stream_adaptive(
            model,
            input_ids,
            chunk_size=chunk_size,
            window_size=window_size,
            sinks=sinks,
            multi_hop=multi_hop,
            n_entities=n_entities,
        )
    raise ValueError(f"unknown mode {mode!r}; use stream|posthoc|full")
