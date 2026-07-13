"""
ByteBudgetKV v0 — training-free prototype.

Idea:
  1) Score prefix tokens from an observation window (SnapKV-style).
  2) Allocate **per-head token budgets** under a global token budget (Ada-KV-style),
     favoring dispersed heads with more slots and sparse heads with fewer.
  3) Store selected KV under a **byte budget lens**: quantize to int8 for
     logical size accounting, dequant to bf16/fp16 for HF decode (no custom kernel yet).

This is the invention scaffold — not yet SOTA; proves the plumbing and metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from snapkv import (
    cache_seq_len,
    compress_recent,
    is_dynamic_cache,
    _pool1d,
)


def _head_concentration(vote: torch.Tensor) -> torch.Tensor:
    """
    vote: [B, H, S] non-negative scores.
    Returns concentration in [0,1] per head (higher = sparser / peakier).
    Uses top-1 mass / total mass.
    """
    total = vote.sum(dim=-1).clamp(min=1e-8)
    top1 = vote.max(dim=-1).values
    return (top1 / total).clamp(0, 1)


def allocate_head_budgets(
    vote: torch.Tensor,
    total_keep: int,
    *,
    min_per_head: int = 8,
    alpha: float = 0.2,
) -> torch.Tensor:
    """
    Ada-KV-inspired budgets from vote [B, H, S] (use batch 0).
    Returns LongTensor [H] of per-head keep counts summing to ~total_keep.
    """
    assert vote.dim() == 3
    b, h, s = vote.shape
    v = vote[0]  # [H, S]
    flat = v.reshape(-1)
    k = min(total_keep * h, flat.numel())  # safety
    # Global top across heads, then count frequency per head
    top_idx = torch.topk(flat, k=min(total_keep, flat.numel()), dim=-1).indices
    head_ids = top_idx // s
    counts = torch.bincount(head_ids, minlength=h).float()
    # Blend with uniform (Ada-KV safeguard alpha)
    uniform = torch.full((h,), total_keep / h, device=vote.device)
    blended = alpha * uniform + (1 - alpha) * counts
    # Normalize to exact total_keep
    blended = blended / blended.sum().clamp(min=1e-8) * total_keep
    budgets = blended.floor().long()
    # Ensure min and fix remainder
    budgets = torch.clamp(budgets, min=min(min_per_head, total_keep // h))
    # Cap per head by s
    budgets = torch.clamp(budgets, max=s)
    diff = int(total_keep - int(budgets.sum().item()))
    # Distribute remainder to least concentrated (most dispersed) heads
    conc = _head_concentration(vote)[0]
    order = torch.argsort(conc)  # low concentration first
    i = 0
    while diff != 0 and i < h * 4:
        hi = int(order[i % h].item())
        if diff > 0 and budgets[hi] < s:
            budgets[hi] += 1
            diff -= 1
        elif diff < 0 and budgets[hi] > 1:
            budgets[hi] -= 1
            diff += 1
        i += 1
    return budgets


def quant_dequant_int8(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Symmetric per-token int8 fake quant.
    Returns (dequantized tensor same dtype as x, logical_int8_nbytes).
    """
    # x: [B, H, S, D] — scale per (B,H,S)
    orig_dtype = x.dtype
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(xf / scale), -128, 127)
    deq = (q * scale).to(orig_dtype)
    # logical storage: int8 payload + fp16 scale per token position
    logical = q.numel() * 1 + scale.numel() * 2
    return deq, logical


def compress_kv_bytebudget(
    past: Any,
    attentions: tuple[torch.Tensor, ...],
    *,
    window_size: int,
    max_capacity: int,
    kernel_size: int = 7,
    use_int8: bool = True,
    sink_size: int = 8,
    expand_radius: int = 2,
    min_per_head: int | None = None,
) -> tuple[Any, dict]:
    """
    Mutates DynamicCache. Returns (past, stats).

    Shared Ada-style token set of size keep_prefix (so seq_len ≈ max_capacity,
    matching SnapKV structure), then optional int8 logical compression
    (dequantized to bf16/fp16 for HF decode).
    """
    from kv_select import attention_to_vote, select_indices_ada

    if not is_dynamic_cache(past):
        raise TypeError(f"Unsupported past type: {type(past)}")
    if max_capacity <= window_size:
        raise ValueError("max_capacity must be > window_size")

    S = cache_seq_len(past)
    stats = {
        "seq_before": S,
        "logical_nbytes": 0,
        "runtime_nbytes": 0,
        "used_int8": use_int8,
    }
    if S <= max_capacity:
        for layer in past.layers:
            if layer.keys is None:
                continue
            stats["runtime_nbytes"] += (
                layer.keys.numel() * layer.keys.element_size()
                + layer.values.numel() * layer.values.element_size()
            )
            stats["logical_nbytes"] = stats["runtime_nbytes"]
        return past, stats

    keep_prefix = max_capacity - window_size
    prefix_len = S - window_size
    logical_total = 0
    runtime_total = 0

    # Floor scales with budget so 512-token runs don't starve mid-context needles
    if min_per_head is None:
        min_per_head = max(16, keep_prefix // 16)

    for layer_idx, layer in enumerate(past.layers):
        k = layer.keys
        v = layer.values
        if k is None:
            continue
        bsz, h_kv, seqlen, head_dim = k.shape

        if (
            layer_idx >= len(attentions)
            or attentions[layer_idx] is None
            or attentions[layer_idx].shape[-1] != S
        ):
            k_new = k[:, :, -max_capacity:, :]
            v_new = v[:, :, -max_capacity:, :]
        else:
            vote = attention_to_vote(
                attentions[layer_idx],
                h_kv=h_kv,
                prefix_len=prefix_len,
                window_size=window_size,
                kernel_size=kernel_size,
                query_power=2.0,
            )
            indices = select_indices_ada(
                vote,
                keep_prefix,
                sink_size=sink_size,
                alpha=0.35,
                expand_radius=expand_radius,
                min_per_head=min_per_head,
            )
            if indices.shape[-1] > keep_prefix:
                indices = indices[:, :, :keep_prefix]
            elif indices.shape[-1] < keep_prefix:
                # pad with last index (should be rare)
                pad_n = keep_prefix - indices.shape[-1]
                indices = torch.cat(
                    [indices, indices[:, :, -1:].expand(-1, -1, pad_n)], dim=-1
                )
            idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_pref = k[:, :, :prefix_len, :]
            v_pref = v[:, :, :prefix_len, :]
            k_sel = torch.gather(k_pref, 2, idx_exp)
            v_sel = torch.gather(v_pref, 2, idx_exp)
            k_new = torch.cat([k_sel, k[:, :, prefix_len:, :]], dim=2)
            v_new = torch.cat([v_sel, v[:, :, prefix_len:, :]], dim=2)

        if use_int8:
            k_new, lnk = quant_dequant_int8(k_new)
            v_new, lnv = quant_dequant_int8(v_new)
            logical_total += lnk + lnv
        else:
            logical_total += (
                k_new.numel() * k_new.element_size()
                + v_new.numel() * v_new.element_size()
            )
        runtime_total += (
            k_new.numel() * k_new.element_size()
            + v_new.numel() * v_new.element_size()
        )
        layer.keys = k_new.contiguous()
        layer.values = v_new.contiguous()

    stats["logical_nbytes"] = logical_total
    stats["runtime_nbytes"] = runtime_total
    stats["seq_after"] = cache_seq_len(past)
    return past, stats


@torch.inference_mode()
def prefill_with_bytebudget(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    window_size: int = 32,
    max_capacity: int = 1024,
    kernel_size: int = 5,
    use_int8: bool = True,
) -> tuple[Any, torch.Tensor]:
    """Returns (past, last_logits). Same full-prefill + score-clone pattern as SnapKV."""
    from snapkv import clone_dynamic_cache, crop_cache_prefix

    seqlen = input_ids.shape[-1]
    if seqlen <= max_capacity:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.logits[:, -1, :]

    out_full = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_full = out_full.past_key_values
    last_logits = out_full.logits[:, -1, :]

    window_size = min(window_size, seqlen - 1)
    prefix_len = seqlen - window_size
    obs_ids = input_ids[:, -window_size:]
    obs_mask = (
        attention_mask[:, -window_size:] if attention_mask is not None else None
    )

    past_score = crop_cache_prefix(clone_dynamic_cache(past_full), prefix_len)
    text_cfg = getattr(model.config, "text_config", model.config)
    old_impl = getattr(text_cfg, "_attn_implementation", None)
    old_root = getattr(model.config, "_attn_implementation", None)
    try:
        if hasattr(text_cfg, "_attn_implementation"):
            text_cfg._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        pos = torch.arange(
            prefix_len, seqlen, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)
        try:
            out_obs = model(
                input_ids=obs_ids,
                attention_mask=obs_mask,
                past_key_values=past_score,
                position_ids=pos,
                cache_position=pos.squeeze(0),
                use_cache=True,
                output_attentions=True,
            )
        except TypeError:
            out_obs = model(
                input_ids=obs_ids,
                attention_mask=obs_mask,
                past_key_values=past_score,
                position_ids=pos,
                use_cache=True,
                output_attentions=True,
            )
    finally:
        if old_impl is not None:
            text_cfg._attn_implementation = old_impl
        if old_root is not None:
            model.config._attn_implementation = old_root

    attns = out_obs.attentions
    if attns is None or all(a is None for a in attns):
        return compress_recent(past_full, max_capacity), last_logits

    past_c, stats = compress_kv_bytebudget(
        past_full,
        attns,
        window_size=window_size,
        max_capacity=max_capacity,
        kernel_size=kernel_size,
        use_int8=use_int8,
    )
    try:
        past_c._bytebudget_stats = stats  # type: ignore[attr-defined]
    except Exception:
        pass
    return past_c, last_logits
