"""
seed_valley: non-oracle shared attention scorer with contiguous segment completion.

Used by bench_scorer_budget / bench_l_epsilon. Training-free.
"""

from __future__ import annotations

from typing import Any

import torch

from kv_select import attention_to_vote
from snapkv import clone_dynamic_cache, crop_cache_prefix, prefill_chunked  # noqa: F401


@torch.inference_mode()
def obs_attentions(
    model,
    input_ids: torch.Tensor,
    past_full,
    window_size: int,
):
    """Eager attn scores for last window tokens over prefix+window."""
    seqlen = int(input_ids.shape[-1])
    window_size = min(window_size, seqlen - 1)
    prefix_len = seqlen - window_size
    obs_ids = input_ids[:, -window_size:]
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
                past_key_values=past_score,
                position_ids=pos,
                cache_position=pos.squeeze(0),
                use_cache=True,
                output_attentions=True,
            )
        except TypeError:
            out_obs = model(
                input_ids=obs_ids,
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

    del past_score
    return out_obs.attentions, window_size, prefix_len


def aggregate_prefix_vote(
    attns: tuple,
    *,
    h_kv: int,
    prefix_len: int,
    window_size: int,
    score_layers: int = 8,
    kernel_size: int = 7,
) -> torch.Tensor:
    usable = [a for a in attns if a is not None]
    if not usable:
        raise RuntimeError("no attentions")
    if score_layers > 0:
        usable = usable[-score_layers:]
    votes = []
    for attn in usable:
        v = attention_to_vote(
            attn,
            h_kv=h_kv,
            prefix_len=prefix_len,
            window_size=window_size,
            kernel_size=kernel_size,
            query_power=2.0,
            use_max=False,
        )
        votes.append(v)
    stacked = torch.stack(votes, dim=0).mean(dim=0)
    return stacked.max(dim=1).values[0]


def grow_valley(
    score_prefix: torch.Tensor,
    seed: int,
    prefix_len: int,
    *,
    floor_ratio: float = 0.25,
    max_radius: int = 24,
) -> set[int]:
    s = float(score_prefix[seed].item())
    thr = max(s * floor_ratio, 0.0)
    lo = hi = seed
    while lo > 0 and seed - lo < max_radius:
        if float(score_prefix[lo - 1].item()) < thr:
            break
        lo -= 1
    while hi + 1 < prefix_len and hi - seed < max_radius:
        if float(score_prefix[hi + 1].item()) < thr:
            break
        hi += 1
    return set(range(lo, hi + 1))


def select_seed_valley(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int = 1,
) -> list[int]:
    budget = min(budget, seq_len)
    recent = list(range(max(0, seq_len - window_size), seq_len))
    sink_idx = list(range(min(sinks, prefix_len, seq_len)))
    forced = set(sink_idx) | set(recent)
    if len(forced) >= budget:
        keep = sorted(set(recent) | set(sink_idx[: max(0, budget - len(recent))]))
        return keep[:budget]

    n_pref = min(prefix_len, int(score_prefix.numel()))
    sc = score_prefix[:n_pref].float().clone()
    for i in sink_idx:
        if i < n_pref:
            sc[i] = -1e9
    order = torch.argsort(sc, descending=True).tolist()
    chosen: set[int] = set(forced)
    covered: set[int] = set()

    for s in order:
        s = int(s)
        if s in covered or s < 0 or s >= n_pref or float(sc[s].item()) <= -1e8:
            continue
        nbhd = grow_valley(score_prefix, s, n_pref)
        for d in range(-expand_radius, expand_radius + 1):
            j = s + d
            if 0 <= j < n_pref:
                nbhd.add(j)
        new = nbhd - chosen
        if len(chosen) + len(new) > budget:
            ranked = sorted(new, key=lambda i: -float(score_prefix[i].item()))
            for j in ranked:
                if len(chosen) >= budget:
                    break
                chosen.add(j)
            covered |= nbhd
            if len(chosen) >= budget:
                break
            continue
        chosen |= new
        covered |= nbhd
        if len(chosen) >= budget:
            break

    if len(chosen) < budget:
        for s in order:
            s = int(s)
            if s in chosen or s < 0 or s >= n_pref:
                continue
            chosen.add(s)
            if len(chosen) >= budget:
                break
    return sorted(chosen)[:budget]


@torch.inference_mode()
def compress_with_seed_valley(
    model,
    input_ids: torch.Tensor,
    past,
    *,
    budget: int,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    score_layers: int = 8,
) -> tuple[Any, list[int]]:
    """
    Score obs-window attention on full past, select seed_valley keep set, compress.
    Mutates past. Returns (past, keep_indices).
    Requires cache_seq_len(past) == input_ids length (full post-hoc path).
    """
    from bench_h1_oracle import compress_keep_indices
    from snapkv import cache_seq_len

    seq_len = int(input_ids.shape[-1])
    if seq_len <= budget:
        return past, list(range(seq_len))
    if cache_seq_len(past) != seq_len:
        raise ValueError(
            "compress_with_seed_valley expects full past; use compress_past_seed_valley"
        )

    attns, window_size, prefix_len = obs_attentions(
        model, input_ids, past, window_size
    )
    if attns is None or all(a is None for a in attns):
        keep = list(range(max(0, seq_len - budget), seq_len))
        past = compress_keep_indices(past, keep)
        return past, keep

    h_kv = past.layers[0].keys.shape[1]
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    keep = select_seed_valley(
        score,
        seq_len=seq_len,
        prefix_len=prefix_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        expand_radius=expand_radius,
    )
    past = compress_keep_indices(past, keep)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return past, keep


@torch.inference_mode()
def obs_attentions_on_past(
    model,
    obs_ids: torch.Tensor,
    past,
    *,
    abs_obs_start: int,
):
    """
    Score last-window queries against current cache.
    past must end with the same tokens as obs_ids (recent always kept).
    abs_obs_start: absolute RoPE index of obs_ids[0].
    """
    from snapkv import cache_seq_len

    window = int(obs_ids.shape[-1])
    S = cache_seq_len(past)
    if S <= window:
        return None, window, 0
    prefix_len = S - window
    past_score = crop_cache_prefix(clone_dynamic_cache(past), prefix_len)

    text_cfg = getattr(model.config, "text_config", model.config)
    old_impl = getattr(text_cfg, "_attn_implementation", None)
    old_root = getattr(model.config, "_attn_implementation", None)
    try:
        if hasattr(text_cfg, "_attn_implementation"):
            text_cfg._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        pos = torch.arange(
            abs_obs_start,
            abs_obs_start + window,
            device=obs_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        try:
            out_obs = model(
                input_ids=obs_ids,
                past_key_values=past_score,
                position_ids=pos,
                cache_position=pos.squeeze(0),
                use_cache=True,
                output_attentions=True,
            )
        except TypeError:
            out_obs = model(
                input_ids=obs_ids,
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

    del past_score
    return out_obs.attentions, window, prefix_len


@torch.inference_mode()
def compress_past_seed_valley(
    model,
    input_ids_so_far: torch.Tensor,
    past,
    *,
    budget: int,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    score_layers: int = 8,
) -> tuple[Any, list[int]]:
    """
    Compress current past (possibly already shorter than prompt) to `budget`.
    Uses last window of input_ids_so_far as observation queries.
    """
    from bench_h1_oracle import compress_keep_indices
    from snapkv import cache_seq_len

    T = int(input_ids_so_far.shape[-1])
    S = cache_seq_len(past)
    if S <= budget:
        return past, list(range(S))

    window_size = min(window_size, T - 1, S - 1)
    if window_size < 1:
        return past, list(range(S))

    obs_ids = input_ids_so_far[:, -window_size:]
    abs_obs_start = T - window_size
    attns, window_size, prefix_len = obs_attentions_on_past(
        model, obs_ids, past, abs_obs_start=abs_obs_start
    )
    if attns is None or all(a is None for a in attns):
        keep = list(range(max(0, S - budget), S))
        past = compress_keep_indices(past, keep)
        return past, keep

    h_kv = past.layers[0].keys.shape[1]
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    # select in *cache* index space (length S)
    keep = select_seed_valley(
        score,
        seq_len=S,
        prefix_len=prefix_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        expand_radius=expand_radius,
    )
    past = compress_keep_indices(past, keep)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return past, keep


@torch.inference_mode()
def prefill_streaming_valley(
    model,
    input_ids: torch.Tensor,
    *,
    stream_budget: int,
    final_budget: int | None = None,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    score_layers: int = 8,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Chunked prefill with online seed_valley compress whenever cache > stream_budget.

    Peak KV stays near stream_budget (+ one chunk), not full L.
    final_budget: optional tighter compress at end (default = stream_budget).

    Returns (past, last_logits, stats).
    """
    from snapkv import cache_nbytes, cache_seq_len

    final_budget = final_budget if final_budget is not None else stream_budget
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device
    past = None
    last_logits = None
    chunk_size = max(int(chunk_size), 1)
    peak_cache = 0
    n_compress = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "position_ids": pos,
        }
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

        peak_cache = max(peak_cache, cache_seq_len(past))

        # Online compress after chunk if over stream budget
        if cache_seq_len(past) > stream_budget:
            past, _ = compress_past_seed_valley(
                model,
                input_ids[:, :end],
                past,
                budget=stream_budget,
                window_size=window_size,
                sinks=sinks,
                expand_radius=expand_radius,
                score_layers=score_layers,
            )
            n_compress += 1
            peak_cache = max(peak_cache, cache_seq_len(past))

    assert past is not None and last_logits is not None

    # Final tighten if requested
    if cache_seq_len(past) > final_budget:
        past, _ = compress_past_seed_valley(
            model,
            input_ids,
            past,
            budget=final_budget,
            window_size=window_size,
            sinks=sinks,
            expand_radius=expand_radius,
            score_layers=score_layers,
        )
        n_compress += 1

    stats = {
        "peak_cache_tokens": peak_cache,
        "final_cache_tokens": cache_seq_len(past),
        "final_kv_mb": round(cache_nbytes(past) / (1024**2), 3),
        "n_compress": n_compress,
        "stream_budget": stream_budget,
        "final_budget": final_budget,
    }
    return past, last_logits, stats
