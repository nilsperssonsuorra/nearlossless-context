"""
SnapKV-style prompt KV compression (training-free).

Compatible with transformers DynamicCache (layers[i].keys / .values).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _pool1d(vote: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """vote: [B, H, S] -> pooled same shape."""
    if kernel_size <= 1:
        return vote
    pad = kernel_size // 2
    b, h, s = vote.shape
    x = vote.reshape(b * h, 1, s)
    x = F.pad(x, (pad, pad), mode="replicate")
    x = F.max_pool1d(x, kernel_size=kernel_size, stride=1)
    return x.reshape(b, h, s)


def is_dynamic_cache(past: Any) -> bool:
    return past is not None and hasattr(past, "layers")


def cache_seq_len(past: Any) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        try:
            return int(past.get_seq_length())
        except Exception:
            pass
    if is_dynamic_cache(past) and past.layers:
        k = past.layers[0].keys
        return 0 if k is None else int(k.shape[-2])
    return 0


def cache_nbytes(past: Any) -> int:
    if not is_dynamic_cache(past):
        return 0
    n = 0
    for layer in past.layers:
        if layer.keys is not None:
            n += layer.keys.numel() * layer.keys.element_size()
        if layer.values is not None:
            n += layer.values.numel() * layer.values.element_size()
    return n


@torch.inference_mode()
def prefill_chunked(
    model,
    input_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
    attention_mask: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor]:
    """
    Prefill in fixed-size chunks to cut peak attention activations (WDDM-friendly).
    Builds full-length DynamicCache; returns (past, last_logits).
    """
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device
    past = None
    last_logits = None
    chunk_size = max(int(chunk_size), 1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
        am = None
        if attention_mask is not None:
            am = attention_mask[:, start:end]
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "position_ids": pos,
        }
        if am is not None:
            kwargs["attention_mask"] = am
        if past is not None:
            kwargs["past_key_values"] = past
            kwargs["cache_position"] = pos.squeeze(0)
        try:
            out = model(**kwargs)
        except TypeError:
            kwargs.pop("cache_position", None)
            out = model(**kwargs)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        del out
    assert past is not None and last_logits is not None
    return past, last_logits


def compress_recent(past: Any, max_capacity: int) -> Any:
    """Keep only the last max_capacity tokens (local window baseline). Mutates cache."""
    if not is_dynamic_cache(past):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    for layer in past.layers:
        if layer.keys is None:
            continue
        s = layer.keys.shape[-2]
        if s > max_capacity:
            layer.keys = layer.keys[:, :, -max_capacity:, :].contiguous()
            layer.values = layer.values[:, :, -max_capacity:, :].contiguous()
    return past


def cache_drop_last(past: Any) -> Any:
    """Drop last sequence position (mutates)."""
    if not is_dynamic_cache(past):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    for layer in past.layers:
        if layer.keys is None:
            continue
        if layer.keys.shape[-2] > 1:
            layer.keys = layer.keys[:, :, :-1, :].contiguous()
            layer.values = layer.values[:, :, :-1, :].contiguous()
    return past


def compress_kv_snapkv(
    past: Any,
    attentions: tuple[torch.Tensor, ...],
    *,
    window_size: int,
    max_capacity: int,
    kernel_size: int = 7,
    sink_size: int = 8,
    expand_radius: int = 2,
) -> Any:
    """
    Mutates DynamicCache in place.
    attentions: from obs-window forward, each [B, H_q, q_len, kv_len]
    """
    from kv_select import attention_to_vote, select_indices_uniform

    if max_capacity <= window_size:
        raise ValueError("max_capacity must be > window_size")
    if not is_dynamic_cache(past):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")

    S = cache_seq_len(past)
    if S <= max_capacity:
        return past

    keep_prefix = max_capacity - window_size
    prefix_len = S - window_size

    for layer_idx, layer in enumerate(past.layers):
        k = layer.keys
        v = layer.values
        if k is None:
            continue
        bsz, h_kv, seqlen, head_dim = k.shape
        if seqlen != S:
            layer.keys = k[:, :, -max_capacity:, :].contiguous()
            layer.values = v[:, :, -max_capacity:, :].contiguous()
            continue

        if layer_idx >= len(attentions) or attentions[layer_idx] is None:
            layer.keys = k[:, :, -max_capacity:, :].contiguous()
            layer.values = v[:, :, -max_capacity:, :].contiguous()
            continue

        attn = attentions[layer_idx]
        if attn.shape[-1] != S:
            layer.keys = k[:, :, -max_capacity:, :].contiguous()
            layer.values = v[:, :, -max_capacity:, :].contiguous()
            continue

        vote = attention_to_vote(
            attn,
            h_kv=h_kv,
            prefix_len=prefix_len,
            window_size=window_size,
            kernel_size=kernel_size,
            query_power=2.0,
            use_max=False,
        )
        k_keep = min(keep_prefix, prefix_len)
        indices = select_indices_uniform(
            vote,
            k_keep,
            sink_size=sink_size,
            expand_radius=expand_radius,
        )
        # Cap width to k_keep after expand
        if indices.shape[-1] > k_keep:
            indices = indices[:, :, :k_keep]

        idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_pref = k[:, :, :prefix_len, :]
        v_pref = v[:, :, :prefix_len, :]
        k_sel = torch.gather(k_pref, dim=2, index=idx_exp)
        v_sel = torch.gather(v_pref, dim=2, index=idx_exp)
        k_obs = k[:, :, prefix_len:, :]
        v_obs = v[:, :, prefix_len:, :]
        layer.keys = torch.cat([k_sel, k_obs], dim=2).contiguous()
        layer.values = torch.cat([v_sel, v_obs], dim=2).contiguous()

    return past


def clone_dynamic_cache(past: Any) -> Any:
    """Deep-copy DynamicCache keys/values into a new DynamicCache."""
    if not is_dynamic_cache(past):
        raise TypeError(type(past))
    from transformers import DynamicCache

    new = DynamicCache()
    for i, layer in enumerate(past.layers):
        if layer.keys is None:
            continue
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


def crop_cache_prefix(past: Any, prefix_len: int) -> Any:
    """Keep only first prefix_len positions (mutates)."""
    for layer in past.layers:
        if layer.keys is None:
            continue
        layer.keys = layer.keys[:, :, :prefix_len, :].contiguous()
        layer.values = layer.values[:, :, :prefix_len, :].contiguous()
    return past


@torch.inference_mode()
def prefill_with_snapkv(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    window_size: int = 32,
    max_capacity: int = 1024,
    kernel_size: int = 5,
) -> tuple[Any, torch.Tensor]:
    """
    Returns (past_key_values, last_prompt_logits [B, vocab]).

    Strategy (more reliable on HF DynamicCache):
      1) Full SDPA prefill once → past + logits
      2) Clone past, crop to prefix, re-forward obs window with eager attn
         only to obtain attention scores
      3) Compress the *original* full past with those scores
    """
    bsz, seqlen = input_ids.shape
    if seqlen <= max_capacity:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.logits[:, -1, :]

    # 1) Prefill (chunked when long — cuts peak SDPA / WDDM thrash)
    if seqlen > 2048:
        past_full, last_logits = prefill_chunked(
            model,
            input_ids,
            chunk_size=512,
            attention_mask=attention_mask,
        )
    else:
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

    # 2) Scoring pass on a clone
    past_score = crop_cache_prefix(clone_dynamic_cache(past_full), prefix_len)

    text_cfg = getattr(model.config, "text_config", model.config)
    old_impl = getattr(text_cfg, "_attn_implementation", None)
    old_root = getattr(model.config, "_attn_implementation", None)
    try:
        if hasattr(text_cfg, "_attn_implementation"):
            text_cfg._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        # Position ids for obs tokens must be prefix_len .. seqlen-1
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

    # 3) Compress original full past
    past_c = compress_kv_snapkv(
        past_full,
        attns,
        window_size=window_size,
        max_capacity=max_capacity,
        kernel_size=kernel_size,
        sink_size=8,
        expand_radius=2,
    )
    return past_c, last_logits
