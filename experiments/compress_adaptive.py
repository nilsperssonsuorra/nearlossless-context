"""
End-to-end adaptive compression: prefill → score → estimate entities → policy → keep.

Posthoc path only for entity estimate (needs full past + question window scores).
Streaming uses L-based policy unless n_entities is provided.
"""

from __future__ import annotations

from typing import Any

import torch

from adaptive import (  # noqa: F401
    AdaptivePolicy,
    model_id_from_model,
    policy_for,
    policy_from_scores,
)
from scorer_valley import (
    aggregate_prefix_vote,
    compress_with_seed_valley,
    obs_attentions,
    prefill_streaming_valley,
)
from snapkv import cache_seq_len, full_layer_h_kv, prefill_chunked


def _resolve_model_id(model, model_id: str | None) -> str | None:
    if model_id:
        return model_id
    mid = model_id_from_model(model)
    return mid or None


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
    h_kv = full_layer_h_kv(past)
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
    use_int8: bool = False,
    model_id: str | None = None,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Chunked full prefill → obs attention → estimate n_entities (optional) →
    adaptive R/budget → seed_valley compress.

    Returns (past, last_logits, info).
    """
    mid = _resolve_model_id(model, model_id)
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
            model_id=mid,
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
            model_id=mid,
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
    logical_mb = None
    if use_int8:
        past, logical = _apply_int8_logical(past)
        logical_mb = round(logical / (1024**2), 3)
    info = {
        "path": "posthoc_adaptive",
        "n_entities_hat": n_hat,
        "policy": pol.__dict__,
        "keep_count": len(keep),
        "cache_tokens": cache_seq_len(past),
        "L": seq_len,
        "logical_kv_mb_int8": logical_mb,
        "use_int8": use_int8,
        "model_id": mid,
    }
    return past, logits, info


def _apply_int8_logical(past) -> tuple[Any, int]:
    """Fake-quant KV to int8 (dequant for HF); return logical nbytes."""
    from bytebudget import quant_dequant_int8

    logical = 0
    for layer in past.layers:
        if layer.keys is None:
            continue
        k, lk = quant_dequant_int8(layer.keys)
        v, lv = quant_dequant_int8(layer.values)
        layer.keys = k.contiguous()
        layer.values = v.contiguous()
        logical += lk + lv
    return past, logical


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
    auto_raise_budget: bool | None = None,
    use_int8: bool = False,
    model_id: str | None = None,
    tokenizer=None,
    discovery: str = "novelty",
) -> tuple[Any, torch.Tensor, dict]:
    """
    Streaming compress with adaptive stream_budget.

    discovery:
      "novelty" — surface novelty pin (query-unknown; recommended default)
      "attn"    — seed_valley mid-stream attention (legacy; needs higher budget)
    """
    mid = _resolve_model_id(model, model_id)
    seq_len = int(input_ids.shape[-1])
    n_ent = n_entities if n_entities is not None else 1
    pol = policy_for(
        n_entities=n_ent,
        L=seq_len,
        multi_hop=multi_hop,
        prefer_stream=True,
        model_id=mid,
    )
    # Attn-only discovery needs larger stream budgets (multi-seed)
    if discovery == "attn" and pol.stream_budget < 1536 and n_ent <= 1 and not multi_hop:
        from dataclasses import replace

        pol = replace(
            pol,
            stream_budget=1536,
            note=pol.note + " | attn-discovery floor 1536",
        )

    if auto_raise_budget is None:
        auto_raise_budget = n_entities is None and discovery == "attn"

    if discovery == "novelty":
        from novelty_detect import prefill_streaming_novelty_pin
        from transformers import AutoTokenizer

        tok = tokenizer
        if tok is None:
            name = mid or getattr(getattr(model, "config", None), "_name_or_path", None)
            if not name:
                raise ValueError(
                    "discovery='novelty' requires tokenizer= or model_id resolvable"
                )
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        past, logits, stats = prefill_streaming_novelty_pin(
            model,
            tok,
            input_ids,
            stream_budget=pol.stream_budget,
            final_budget=pol.stream_budget,
            chunk_size=chunk_size,
            window_size=window_size,
            sinks=sinks,
            expand_radius=pol.R,
        )
        path = "stream_novelty"
    else:
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
            auto_raise_budget=auto_raise_budget,
            multi_budget_floor=max(1024, pol.stream_budget),
        )
        path = "stream_adaptive_attn"

    logical_mb = None
    if use_int8:
        past, logical = _apply_int8_logical(past)
        logical_mb = round(logical / (1024**2), 3)
    info = {
        "path": path,
        "discovery": discovery,
        "n_entities_hat": stats.get("n_entities_hat", n_ent),
        "policy": pol.__dict__,
        "stats": stats,
        "L": seq_len,
        "logical_kv_mb_int8": logical_mb,
        "use_int8": use_int8,
        "model_id": mid,
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
    safe_multi: bool = False,
    use_int8: bool = False,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    model_id: str | None = None,
    tokenizer=None,
    discovery: str = "novelty",
) -> tuple[Any, torch.Tensor, dict]:
    """
    One-call entrypoint for “fit more context under 24GB”.

    mode:
      "stream"  — online compress (low peak cache; default for long L)
      "posthoc" — full prefill then compress (auto n̂ if n_entities is None)
      "full"    — chunked full KV (no compress)

    discovery (stream only):
      "novelty" — query-unknown surface novelty pin (default; multi-seed strong)
      "attn"    — mid-stream seed_valley attention (needs larger budgets)

    Recommended:
      long single-doc retrieval → mode="stream" (omit n_entities)
      multi-secret / multi-doc → mode="stream", n_entities=3 or safe_multi=True
      best quality post-score → mode="posthoc"

    safe_multi: if True and n_entities is None, use n_entities=3 schedule.
    use_int8: fake-quant final KV for ~2× logical byte accounting (HF still dequants).
    model_id: optional HF id; if omitted, taken from model.config for family floors.
    tokenizer: required for discovery="novelty" unless model_id can load one.
    """
    mid = _resolve_model_id(model, model_id)
    if safe_multi and n_entities is None:
        n_entities = 3
    if mode == "full":
        past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
        info = {
            "path": "full_chunked",
            "L": int(input_ids.shape[-1]),
            "cache_tokens": cache_seq_len(past),
            "policy": None,
            "model_id": mid,
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
            use_int8=use_int8,
            model_id=mid,
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
            use_int8=use_int8,
            model_id=mid,
            tokenizer=tokenizer,
            discovery=discovery,
        )
    raise ValueError(f"unknown mode {mode!r}; use stream|posthoc|full")
