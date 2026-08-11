"""
seed_valley: non-oracle shared attention scorer with contiguous segment completion.

Used by bench_scorer_budget / bench_l_epsilon. Training-free.
"""

from __future__ import annotations

from typing import Any

import torch

if __package__:
    from .kv_select import attention_to_vote
    from .snapkv import clone_dynamic_cache, crop_cache_prefix, prefill_chunked  # noqa: F401
else:
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
    """
    Aggregate obs-window attention into a prefix score vector [prefix_len].

    Hybrid models emit sliding + full attentions with different kv_len; only
    scores with kv_len == prefix_len + window_size (full-attention path) are used.
    """
    target_kv = int(prefix_len) + int(window_size)
    usable = [
        a
        for a in attns
        if a is not None and int(a.shape[-1]) == target_kv
    ]
    if not usable:
        # Fallback: any attn whose kv covers at least the prefix
        usable = [
            a
            for a in attns
            if a is not None and int(a.shape[-1]) >= int(prefix_len)
        ]
    if not usable:
        raise RuntimeError("no attentions")
    if score_layers > 0:
        usable = usable[-score_layers:]
    votes = []
    for attn in usable:
        kv_len = int(attn.shape[-1])
        # If kv_len matches full path, standard vote; else vote over available prefix slice
        pl = min(int(prefix_len), kv_len)
        v = attention_to_vote(
            attn,
            h_kv=h_kv,
            prefix_len=pl,
            window_size=min(int(window_size), max(kv_len - pl, 0)),
            kernel_size=kernel_size,
            query_power=2.0,
            use_max=False,
        )
        if pl < int(prefix_len):
            # Pad left so indices align to absolute prefix (early tokens missing)
            pad = torch.zeros(
                *v.shape[:-1],
                int(prefix_len) - pl,
                device=v.device,
                dtype=v.dtype,
            )
            v = torch.cat([pad, v], dim=-1)
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


def find_local_maxima(
    score_prefix: torch.Tensor,
    prefix_len: int,
    *,
    min_score: float | None = None,
) -> list[int]:
    """Strict local peaks (plateaus: keep leftmost max)."""
    n = min(prefix_len, int(score_prefix.numel()))
    if n <= 0:
        return []
    sc = score_prefix[:n]
    if min_score is None:
        # keep peaks above median of positive-ish mass
        min_score = float(sc.median().item())
    peaks: list[int] = []
    for i in range(n):
        v = float(sc[i].item())
        if v < min_score or v <= -1e8:
            continue
        left = float(sc[i - 1].item()) if i > 0 else -1e30
        right = float(sc[i + 1].item()) if i + 1 < n else -1e30
        if v >= left and v > right:
            peaks.append(i)
        elif v > left and v >= right:
            peaks.append(i)
    return peaks


def select_seed_valley_multipeak(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int = 1,
    min_sep: int = 64,
    max_peaks: int = 16,
) -> list[int]:
    """
    Multi-needle aware selection: cover diverse local maxima first (separated
    by min_sep), grow each valley, then fill remainder with vanilla seed_valley.
    """
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

    # Local maxima ranked by score
    peaks = find_local_maxima(score_prefix, n_pref)
    peaks = [p for p in peaks if float(sc[p].item()) > -1e8]
    peaks.sort(key=lambda i: float(score_prefix[i].item()), reverse=True)

    chosen: set[int] = set(forced)
    seed_centers: list[int] = []
    covered: set[int] = set()

    def add_seed(s: int) -> bool:
        nonlocal chosen, covered
        if s < 0 or s >= n_pref:
            return False
        if any(abs(s - c) < min_sep for c in seed_centers):
            return False
        nbhd = grow_valley(score_prefix, s, n_pref)
        for d in range(-expand_radius, expand_radius + 1):
            j = s + d
            if 0 <= j < n_pref:
                nbhd.add(j)
        new = nbhd - chosen
        if not new and s in chosen:
            seed_centers.append(s)
            covered |= nbhd
            return True
        if len(chosen) + len(new) > budget:
            ranked = sorted(new, key=lambda i: -float(score_prefix[i].item()))
            for j in ranked:
                if len(chosen) >= budget:
                    break
                chosen.add(j)
            seed_centers.append(s)
            covered |= nbhd
            return len(chosen) >= budget
        chosen |= new
        seed_centers.append(s)
        covered |= nbhd
        return len(chosen) >= budget

    for p in peaks:
        if len(seed_centers) >= max_peaks:
            break
        if len(chosen) >= budget:
            break
        add_seed(int(p))

    # Also diversify over global top-k if few peaks found
    if len(seed_centers) < max(3, max_peaks // 2) and len(chosen) < budget:
        order = torch.argsort(sc, descending=True).tolist()
        for s in order:
            if len(seed_centers) >= max_peaks or len(chosen) >= budget:
                break
            s = int(s)
            if float(sc[s].item()) <= -1e8:
                continue
            add_seed(s)

    # Fill rest with dense seed_valley (no sep constraint)
    if len(chosen) < budget:
        rest = select_seed_valley(
            score_prefix,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
        )
        for i in rest:
            if len(chosen) >= budget:
                break
            chosen.add(i)

    return sorted(chosen)[:budget]


def select_seed_valley_binned(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int = 1,
    n_bins: int = 8,
    seeds_per_bin: int = 2,
) -> list[int]:
    """
    Spatial coverage: take top seeds inside each depth bin, grow valleys.
    Guarantees multi-needle positions are not all stolen by one high-attn region.
    """
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

    n_bins = max(1, min(n_bins, n_pref))
    bin_size = max(1, (n_pref + n_bins - 1) // n_bins)
    chosen: set[int] = set(forced)

    for b in range(n_bins):
        lo = b * bin_size
        hi = min(n_pref, (b + 1) * bin_size)
        if lo >= hi:
            continue
        seg = sc[lo:hi]
        k = min(seeds_per_bin, hi - lo)
        top = torch.topk(seg, k=k).indices.tolist()
        for ti in top:
            s = lo + int(ti)
            if float(sc[s].item()) <= -1e8:
                continue
            nbhd = grow_valley(score_prefix, s, n_pref)
            for d in range(-expand_radius, expand_radius + 1):
                j = s + d
                if 0 <= j < n_pref:
                    nbhd.add(j)
            new = nbhd - chosen
            if len(chosen) + len(new) <= budget:
                chosen |= new
            else:
                for j in sorted(new, key=lambda i: -float(score_prefix[i].item())):
                    if len(chosen) >= budget:
                        break
                    chosen.add(j)
            if len(chosen) >= budget:
                break
        if len(chosen) >= budget:
            break

    if len(chosen) < budget:
        rest = select_seed_valley(
            score_prefix,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
        )
        for i in rest:
            if len(chosen) >= budget:
                break
            chosen.add(i)
    return sorted(chosen)[:budget]


def _select_keep(
    score: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int,
    mode: str,
    min_sep: int,
    max_peaks: int,
) -> list[int]:
    if mode == "multipeak":
        return select_seed_valley_multipeak(
            score,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
            min_sep=min_sep,
            max_peaks=max_peaks,
        )
    if mode == "binned":
        return select_seed_valley_binned(
            score,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
            n_bins=max(4, max_peaks // 2),
            seeds_per_bin=2,
        )
    return select_seed_valley(
        score,
        seq_len=seq_len,
        prefix_len=prefix_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        expand_radius=expand_radius,
    )


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
    mode: str = "valley",
    min_sep: int = 64,
    max_peaks: int = 16,
) -> tuple[Any, list[int]]:
    """
    Score obs-window attention on full past, select seed_valley keep set, compress.
    Mutates past. Returns (past, keep_indices).
    Requires cache_seq_len(past) == input_ids length (full post-hoc path).
    mode: "valley" | "multipeak"
    """
    if __package__:
        from .snapkv import cache_seq_len, compress_keep_indices
    else:
        from snapkv import cache_seq_len, compress_keep_indices

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

    if __package__:
        from .snapkv import full_layer_h_kv
    else:
        from snapkv import full_layer_h_kv

    h_kv = full_layer_h_kv(past)
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    keep = _select_keep(
        score,
        seq_len=seq_len,
        prefix_len=prefix_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        expand_radius=expand_radius,
        mode=mode,
        min_sep=min_sep,
        max_peaks=max_peaks,
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
    if __package__:
        from .snapkv import cache_seq_len
    else:
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
    mode: str = "valley",
    min_sep: int = 64,
    max_peaks: int = 16,
) -> tuple[Any, list[int]]:
    """
    Compress current past (possibly already shorter than prompt) to `budget`.
    Uses last window of input_ids_so_far as observation queries.
    """
    if __package__:
        from .snapkv import cache_seq_len, compress_keep_indices
    else:
        from snapkv import cache_seq_len, compress_keep_indices

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

    if __package__:
        from .snapkv import full_layer_h_kv
    else:
        from snapkv import full_layer_h_kv

    h_kv = full_layer_h_kv(past)
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    # select in *cache* index space (length S)
    keep = _select_keep(
        score,
        seq_len=S,
        prefix_len=prefix_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        expand_radius=expand_radius,
        mode=mode,
        min_sep=min_sep,
        max_peaks=max_peaks,
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
    mode: str = "valley",
    min_sep: int = 64,
    max_peaks: int = 16,
    auto_raise_budget: bool = False,
    multi_budget_floor: int = 1024,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Chunked prefill with online seed_valley compress whenever cache > stream_budget.

    Peak KV stays near stream_budget (+ one chunk), not full L.
    final_budget: optional tighter compress at end (default = stream_budget).

    auto_raise_budget: before first hard compress, estimate n_entities from
    mid-stream scores; if ≥2, raise stream_budget floor to multi_budget_floor
    (and bump expand_radius to at least 8). Must run *before* dropping spans.

    Returns (past, last_logits, stats).
    """
    if __package__:
        from .adaptive import estimate_n_entities_from_scores
        from .snapkv import cache_nbytes, cache_seq_len
    else:
        from adaptive import estimate_n_entities_from_scores
        from snapkv import cache_nbytes, cache_seq_len

    dyn_budget = int(stream_budget)
    dyn_R = int(expand_radius)
    final_budget = final_budget if final_budget is not None else dyn_budget
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device
    past = None
    last_logits = None
    chunk_size = max(int(chunk_size), 1)
    peak_cache = 0
    n_compress = 0
    n_hat = None
    raised = False
    probe_at = min(max(dyn_budget, 512), max(seq_len // 4, 512))

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

        # Mid-stream entity estimate before first aggressive drop
        if (
            auto_raise_budget
            and not raised
            and past is not None
            and cache_seq_len(past) >= probe_at
        ):
            try:
                S = cache_seq_len(past)
                w = min(window_size, S - 1, end - 1)
                if w >= 8:
                    obs_ids = input_ids[:, end - w : end]
                    attns, w, pref = obs_attentions_on_past(
                        model, obs_ids, past, abs_obs_start=end - w
                    )
                    if attns is not None and any(a is not None for a in attns):
                        h_kv = past.layers[0].keys.shape[1]
                        score = aggregate_prefix_vote(
                            attns,
                            h_kv=h_kv,
                            prefix_len=pref,
                            window_size=w,
                            score_layers=score_layers,
                        )
                        n_hat = estimate_n_entities_from_scores(
                            score, prefix_len=pref
                        )
                        if n_hat >= 2:
                            dyn_budget = max(dyn_budget, multi_budget_floor)
                            dyn_R = max(dyn_R, 8)
                            final_budget = max(final_budget, dyn_budget)
                        raised = True
            except Exception:
                raised = True  # don't retry forever

        # Online compress after chunk if over stream budget
        if cache_seq_len(past) > dyn_budget:
            past, _ = compress_past_seed_valley(
                model,
                input_ids[:, :end],
                past,
                budget=dyn_budget,
                window_size=window_size,
                sinks=sinks,
                expand_radius=dyn_R,
                score_layers=score_layers,
                mode=mode,
                min_sep=min_sep,
                max_peaks=max_peaks,
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
            expand_radius=dyn_R,
            score_layers=score_layers,
            mode=mode,
            min_sep=min_sep,
            max_peaks=max_peaks,
        )
        n_compress += 1

    stats = {
        "peak_cache_tokens": peak_cache,
        "final_cache_tokens": cache_seq_len(past),
        "final_kv_mb": round(cache_nbytes(past) / (1024**2), 3),
        "n_compress": n_compress,
        "stream_budget": dyn_budget,
        "final_budget": final_budget,
        "mode": mode,
        "n_entities_hat": n_hat,
        "auto_raise_budget": auto_raise_budget,
        "expand_radius": dyn_R,
    }
    return past, last_logits, stats
