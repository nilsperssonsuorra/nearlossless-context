"""Shared prefill + greedy decode for compression methods."""

from __future__ import annotations

from typing import Any

import torch

from snapkv import compress_recent, prefill_with_snapkv


def _sanitize_mask(
    attention_mask: torch.Tensor | None,
    input_ids: torch.Tensor,
) -> torch.Tensor | None:
    """All-ones masks can break some Qwen+compressed-cache paths; treat as None."""
    if attention_mask is None:
        return None
    if torch.all(attention_mask == 1):
        return None
    return attention_mask


@torch.inference_mode()
def prefill_method(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    method: str,
    *,
    budget: int = 1024,
    window: int = 32,
    kernel: int = 5,
) -> tuple[Any, torch.Tensor]:
    """Return (past_key_values, last_prompt_logits)."""
    attention_mask = _sanitize_mask(attention_mask, input_ids)
    if method == "full":
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return out.past_key_values, out.logits[:, -1, :]
    if method == "recent":
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past = compress_recent(out.past_key_values, budget)
        return past, out.logits[:, -1, :]
    if method == "snapkv":
        return prefill_with_snapkv(
            model,
            input_ids,
            attention_mask,
            window_size=window,
            max_capacity=budget,
            kernel_size=kernel,
        )
    if method == "bytebudget":
        from bytebudget import prefill_with_bytebudget

        return prefill_with_bytebudget(
            model,
            input_ids,
            attention_mask,
            window_size=window,
            max_capacity=budget,
            kernel_size=kernel,
        )
    raise ValueError(f"Unknown method: {method}")


@torch.inference_mode()
def greedy_generate(
    model,
    past: Any,
    last_logits: torch.Tensor,
    n_new: int,
    *,
    eos_id: int | None = None,
    next_position: int | None = None,
) -> list[int]:
    """
    Greedy decode n_new tokens.

    next_position: absolute position index for the *first generated* token
    (normally = original prompt length). Critical when past was compressed:
    RoPE on new queries must continue from the true sequence index, not
    len(compressed_cache).
    """
    tokens: list[int] = []
    device = last_logits.device
    cur = torch.argmax(last_logits, dim=-1, keepdim=True)  # [B,1]
    tokens.append(int(cur.item()))

    # Position for the token we are about to feed was the last prompt token
    # when computing last_logits; the *next* forward embeds `cur` at position
    # = next_position (first new token index).
    if next_position is None:
        # Fallback: trust cache length (OK for full KV only)
        if hasattr(past, "get_seq_length"):
            pos = int(past.get_seq_length())
        else:
            pos = 0
    else:
        pos = int(next_position)

    for i in range(n_new - 1):
        if eos_id is not None and tokens[-1] == eos_id:
            break
        position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
        cache_position = torch.tensor([pos], device=device, dtype=torch.long)
        try:
            o = model(
                input_ids=cur,
                past_key_values=past,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
            )
        except TypeError:
            o = model(
                input_ids=cur,
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True,
            )
        past = o.past_key_values
        cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        tokens.append(int(cur.item()))
        pos += 1
    return tokens
