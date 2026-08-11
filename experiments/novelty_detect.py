"""
Query-unknown fact discovery via surface novelty (no final question).

Motivation: oracle-online stream@512 hits 15/15 multi-seed; scored attention
capsules do not. Mid-stream attention is a weak detector. Facts often look
*different* from repetitive filler: rare tokens, digits, hyphenated IDs.

This module scores absolute positions by local novelty and proposes capsules
(critical±R style neighborhoods around novelty peaks).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import torch

if __package__:
    from .capsules import FactCapsule, _components_to_capsules
else:
    from capsules import FactCapsule, _components_to_capsules


_DIGIT_RE = re.compile(r"\d")
_IDISH_RE = re.compile(r"[A-Za-z]+[-_][A-Za-z0-9]+|[A-Z]{2,}[-_]?[0-9]+|[0-9]+[-_][A-Za-z]+")


def _token_surface_features(tokenizer, token_id: int) -> dict[str, float]:
    try:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        piece = ""
    piece = piece or ""
    has_digit = 1.0 if _DIGIT_RE.search(piece) else 0.0
    idish = 1.0 if _IDISH_RE.search(piece) else 0.0
    # mixed case / code-like
    has_upper = any(c.isupper() for c in piece)
    has_lower = any(c.islower() for c in piece)
    mixed = 1.0 if has_upper and (has_lower or has_digit) else 0.0
    # non-alpha fraction
    if piece:
        non_alnum = sum(1 for c in piece if not c.isalnum() and not c.isspace()) / max(
            len(piece), 1
        )
    else:
        non_alnum = 0.0
    return {
        "digit": has_digit,
        "idish": idish,
        "mixed": mixed,
        "punct": float(non_alnum),
        "len": float(min(len(piece), 16)),
    }


def score_novelty(
    tokenizer,
    input_ids: list[int] | torch.Tensor,
    *,
    smooth_window: int = 3,
) -> list[float]:
    """
    Per-position novelty score for a full or prefix sequence.

    Combines inverse document frequency (within-prefix), first-occurrence,
    and surface cues (digits, ID-like pieces).
    """
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.view(-1).tolist()
    else:
        ids = list(input_ids)
    n = len(ids)
    if n == 0:
        return []

    counts = Counter(ids)
    first_idx: dict[int, int] = {}
    for i, t in enumerate(ids):
        if t not in first_idx:
            first_idx[t] = i

    # Cache surface features by token id
    feat_cache: dict[int, dict[str, float]] = {}
    raw = [0.0] * n
    for i, t in enumerate(ids):
        if t not in feat_cache:
            feat_cache[t] = _token_surface_features(tokenizer, t)
        f = feat_cache[t]
        # rarity: high when token is rare in this prefix
        rare = math.log(1.0 + n / max(counts[t], 1))
        first = 1.0 if first_idx[t] == i else 0.0
        # Downweight pure whitespace/special-looking short tokens
        surface = (
            2.5 * f["digit"]
            + 2.0 * f["idish"]
            + 1.0 * f["mixed"]
            + 0.8 * f["punct"]
        )
        raw[i] = rare + 0.75 * first + surface

    # Local max-pool smooth so multi-token IDs form a peak region
    if smooth_window <= 1:
        return raw
    half = smooth_window // 2
    smooth = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smooth[i] = max(raw[lo:hi])
    return smooth


def discover_novelty_capsules(
    tokenizer,
    input_ids: list[int] | torch.Tensor,
    *,
    expand_radius: int = 1,
    max_capsules: int = 12,
    min_sep: int = 32,
    floor_ratio: float = 0.55,
    abs_offset: int = 0,
    skip_prefix: int = 8,
    skip_suffix: int = 8,
) -> list[FactCapsule]:
    """
    Propose absolute-position capsules from high-novelty regions.

    Threshold all positions ≥ floor_ratio * max, expand ±R, take connected
    components ranked by peak score (no min_sep between peaks — multi-token
    IDs / nearby secrets would otherwise kill each other).

    skip_prefix/suffix: ignore chat template / trailing specials.
    """
    del min_sep  # kept for API compat; components replace peak spacing
    scores = score_novelty(tokenizer, input_ids)
    n = len(scores)
    if n == 0:
        return []
    lo_i = min(skip_prefix, n)
    hi_i = max(lo_i + 1, n - skip_suffix)
    mid = scores[lo_i:hi_i]
    gmax = max(mid) if mid else max(scores)
    if gmax <= 0:
        return []
    thr = gmax * floor_ratio

    score_map: dict[int, float] = {
        abs_offset + i: scores[i] for i in range(n)
    }
    marked: set[int] = set()
    for i in range(lo_i, hi_i):
        if scores[i] < thr:
            continue
        abs_i = abs_offset + i
        for d in range(-expand_radius, expand_radius + 1):
            j = abs_i + d
            if abs_offset + lo_i <= j < abs_offset + hi_i:
                marked.add(j)

    caps = _components_to_capsules(
        marked, score_map, source="novelty", score_floor=thr
    )
    # Rank components by peak score; keep top max_capsules
    caps.sort(key=lambda c: -c.score)
    return caps[:max_capsules]


def novelty_abs_set(
    tokenizer,
    input_ids: list[int] | torch.Tensor,
    *,
    expand_radius: int = 1,
    max_capsules: int = 12,
    min_sep: int = 32,
    floor_ratio: float = 0.65,
) -> set[int]:
    """Union of absolute positions covered by novelty capsules."""
    caps = discover_novelty_capsules(
        tokenizer,
        input_ids,
        expand_radius=expand_radius,
        max_capsules=max_capsules,
        min_sep=min_sep,
        floor_ratio=floor_ratio,
        abs_offset=0,
    )
    out: set[int] = set()
    for c in caps:
        out.update(range(c.lo, c.hi + 1))
    return out


@torch.inference_mode()
def prefill_streaming_novelty_pin(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    stream_budget: int,
    final_budget: int | None = None,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    max_capsules: int = 12,
    min_sep: int = 32,
    floor_ratio: float = 0.65,
    sticky: bool = True,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Stream compress forcing keep of *novelty* capsules (query-unknown detector).

    Discovery = surface novelty on the prefix so far. Long-L stability:
      - max_capsules scales with prefix length (more false peaks at large L)
      - sticky pin registry: once discovered, abs positions stay pinned until
        evicted (avoids re-rank thrash dropping a previously found secret)
      - pin packing ranked by novelty score (not cache order) under budget
    """
    if __package__:
        from .snapkv import cache_seq_len, compress_keep_indices
    else:
        from snapkv import cache_seq_len, compress_keep_indices

    dyn_budget = int(stream_budget)
    final_budget = int(final_budget if final_budget is not None else dyn_budget)
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device
    past = None
    last_logits = None
    chunk_size = max(int(chunk_size), 1)
    peak_cache = 0
    n_compress = 0
    abs_pos: list[int] = []
    last_n_novelty = 0
    last_recall_proxy = 0.0
    sticky_pins: set[int] = set()
    last_scores: list[float] = []
    dyn_max_caps = int(max_capsules)

    def _caps_for_len(end: int) -> int:
        # Grow discovery capacity slowly with L (12 @4k → ~32 @24k → 48 cap)
        return max(int(max_capsules), min(48, 8 + end // 1024))

    def _discover_set(end: int) -> tuple[set[int], list[float]]:
        nonlocal dyn_max_caps
        ids_list = input_ids[0, :end].tolist()
        scores = score_novelty(tokenizer, ids_list)
        dyn_max_caps = _caps_for_len(end)
        fresh = novelty_abs_set(
            tokenizer,
            ids_list,
            expand_radius=expand_radius,
            max_capsules=dyn_max_caps,
            min_sep=min_sep,
            floor_ratio=floor_ratio,
        )
        if sticky:
            sticky_pins.update(fresh)
            pin = set(sticky_pins)
        else:
            pin = set(fresh)
        return pin, scores

    def _compress(
        past_kv,
        abs_list: list[int],
        budget: int,
        pin_abs: set[int],
        scores: list[float],
    ):
        nonlocal last_n_novelty, last_recall_proxy
        S = cache_seq_len(past_kv)
        if S <= budget:
            return past_kv, abs_list
        recent = list(range(max(0, S - window_size), S))
        sink_idx = list(range(min(sinks, S)))
        # Reserve sinks+recent first so pins cannot starve the question window
        keep: list[int] = []
        seen: set[int] = set()
        for i in sink_idx + recent:
            if 0 <= i < S and i not in seen:
                keep.append(i)
                seen.add(i)
        reserve = len(seen)
        pin_slots = max(0, budget - reserve)

        # Rank pin hits still in cache by novelty score (high first)
        pin_hits: list[tuple[float, int]] = []
        for i, a in enumerate(abs_list):
            if i in seen or a not in pin_abs:
                continue
            sc = scores[a] if 0 <= a < len(scores) else 0.0
            pin_hits.append((sc, i))
        pin_hits.sort(key=lambda t: (-t[0], t[1]))
        n_pin_kept = 0
        for _, i in pin_hits:
            if n_pin_kept >= pin_slots:
                break
            if i not in seen:
                keep.append(i)
                seen.add(i)
                n_pin_kept += 1
        last_n_novelty = n_pin_kept

        # Fill remainder by cache order
        for i in range(S):
            if len(keep) >= budget:
                break
            if i not in seen:
                keep.append(i)
                seen.add(i)
        keep = sorted(keep[:budget])
        past_kv = compress_keep_indices(past_kv, keep)
        abs_list = [abs_list[i] for i in keep]
        still = sum(1 for a in abs_list if a in pin_abs)
        last_recall_proxy = still / max(len(pin_abs), 1)
        # Sticky only tracks pins still resident in cache
        if sticky:
            sticky_pins.intersection_update(abs_list)
        return past_kv, abs_list

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

        abs_pos.extend(range(start, end))
        S_now = cache_seq_len(past)
        if len(abs_pos) > S_now:
            abs_pos = abs_pos[-S_now:]
        peak_cache = max(peak_cache, S_now)

        if past is not None and S_now > dyn_budget:
            pin_abs, last_scores = _discover_set(end)
            past, abs_pos = _compress(
                past, abs_pos, dyn_budget, pin_abs, last_scores
            )
            n_compress += 1
            peak_cache = max(peak_cache, cache_seq_len(past))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert past is not None and last_logits is not None
    if cache_seq_len(past) > final_budget:
        pin_abs, last_scores = _discover_set(seq_len)
        past, abs_pos = _compress(
            past, abs_pos, final_budget, pin_abs, last_scores
        )
        n_compress += 1

    stats = {
        "path": "stream_novelty_pin",
        "stream_budget": dyn_budget,
        "final_budget": final_budget,
        "peak_cache": peak_cache,
        "final_cache": cache_seq_len(past),
        "n_compress": n_compress,
        "last_n_novelty_kept": last_n_novelty,
        "last_novelty_retention_proxy": round(last_recall_proxy, 4),
        "floor_ratio": floor_ratio,
        "max_capsules": dyn_max_caps,
        "sticky": sticky,
        "n_sticky_pins": len(sticky_pins),
    }
    return past, last_logits, stats


def _attention_peak_abs(
    model,
    input_ids_so_far: torch.Tensor,
    past_kv,
    abs_list: list[int],
    *,
    window_size: int = 128,
    expand_radius: int = 1,
    max_peaks: int = 12,
    score_layers: int = 8,
) -> set[int]:
    """
    Map mid/query-window attention peaks in *cache* index space to absolute
    positions via abs_list. Used for hybrid discovery.
    """
    if __package__:
        from .scorer_valley import (
            aggregate_prefix_vote,
            find_local_maxima,
            obs_attentions_on_past,
        )
        from .snapkv import cache_seq_len, full_layer_h_kv
    else:
        from scorer_valley import (
            aggregate_prefix_vote,
            find_local_maxima,
            obs_attentions_on_past,
        )
        from snapkv import cache_seq_len, full_layer_h_kv

    T = int(input_ids_so_far.shape[-1])
    S = cache_seq_len(past_kv)
    if S <= 1 or not abs_list:
        return set()
    w = min(window_size, T - 1, S - 1)
    if w < 4:
        return set()
    obs_ids = input_ids_so_far[:, -w:]
    abs_obs_start = T - w
    try:
        attns, w, prefix_len = obs_attentions_on_past(
            model, obs_ids, past_kv, abs_obs_start=abs_obs_start
        )
    except Exception:
        return set()
    if attns is None or all(a is None for a in attns):
        return set()
    try:
        h_kv = full_layer_h_kv(past_kv)
        score = aggregate_prefix_vote(
            attns,
            h_kv=h_kv,
            prefix_len=prefix_len,
            window_size=w,
            score_layers=score_layers,
        )
    except Exception:
        return set()
    n_pref = min(prefix_len, int(score.numel()), S, len(abs_list))
    if n_pref <= 0:
        return set()
    # Mask sinks lightly so peaks come from content
    sc = score[:n_pref].float().clone()
    for i in range(min(8, n_pref)):
        sc[i] = -1e9
    peaks = find_local_maxima(sc, n_pref, min_score=float(sc.median().item()))
    peaks = sorted(peaks, key=lambda i: -float(sc[i].item()))[:max_peaks]
    if not peaks:
        # fallback: top-k raw scores
        order = torch.argsort(sc, descending=True).tolist()[:max_peaks]
        peaks = [int(i) for i in order if float(sc[int(i)].item()) > -1e8]
    out: set[int] = set()
    for i in peaks:
        if i < 0 or i >= len(abs_list):
            continue
        a = abs_list[i]
        for d in range(-expand_radius, expand_radius + 1):
            out.add(a + d)
    return out


@torch.inference_mode()
def prefill_streaming_hybrid_pin(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    stream_budget: int,
    final_budget: int | None = None,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    max_capsules: int = 12,
    min_sep: int = 32,
    floor_ratio: float = 0.65,
    sticky: bool = True,
    score_layers: int = 8,
    attn_max_peaks: int = 16,
    hold_factor: float = 1.5,
    hold_budget: int | None = None,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Hybrid stream: sticky surface novelty ∪ mid-stream attention peaks online;
    final compress is *query-aware* (last window ≈ question) re-ranking while
    still prioritizing novelty pins that survived.

    hold_budget: absolute mid-stream cache cap (preferred). If None, derive from
    hold_factor * final_budget (capped +512). Larger hold keeps more tokens until
    the question arrives so query-aware re-rank has survivors to promote.
    Peak cache ≈ hold_budget + chunk_size.
    """
    if __package__:
        from .snapkv import cache_seq_len, compress_keep_indices
    else:
        from snapkv import cache_seq_len, compress_keep_indices

    final_budget = int(final_budget if final_budget is not None else stream_budget)
    if hold_budget is not None:
        hold_budget = int(max(final_budget, hold_budget))
    else:
        hold_budget = int(
            max(final_budget, min(int(final_budget * hold_factor), final_budget + 512))
        )
    dyn_budget = hold_budget
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device
    past = None
    last_logits = None
    chunk_size = max(int(chunk_size), 1)
    peak_cache = 0
    n_compress = 0
    abs_pos: list[int] = []
    last_n_novelty = 0
    last_n_attn_pins = 0
    last_recall_proxy = 0.0
    sticky_pins: set[int] = set()
    last_scores: list[float] = []
    dyn_max_caps = int(max_capsules)

    def _caps_for_len(end: int) -> int:
        return max(int(max_capsules), min(48, 8 + end // 1024))

    def _novelty_set(end: int) -> tuple[set[int], list[float]]:
        nonlocal dyn_max_caps
        ids_list = input_ids[0, :end].tolist()
        scores = score_novelty(tokenizer, ids_list)
        dyn_max_caps = _caps_for_len(end)
        fresh = novelty_abs_set(
            tokenizer,
            ids_list,
            expand_radius=expand_radius,
            max_capsules=dyn_max_caps,
            min_sep=min_sep,
            floor_ratio=floor_ratio,
        )
        if sticky:
            sticky_pins.update(fresh)
            pin = set(sticky_pins)
        else:
            pin = set(fresh)
        return pin, scores

    def _compress(
        past_kv,
        abs_list: list[int],
        budget: int,
        pin_abs: set[int],
        scores: list[float],
        attn_rank: list[float] | None = None,
    ):
        """
        Pack priority: sinks+recent → novelty pins (by novelty score) →
        remaining by attention rank if provided else cache order.
        """
        nonlocal last_n_novelty, last_recall_proxy
        S = cache_seq_len(past_kv)
        if S <= budget:
            return past_kv, abs_list
        recent = list(range(max(0, S - window_size), S))
        sink_idx = list(range(min(sinks, S)))
        keep: list[int] = []
        seen: set[int] = set()
        for i in sink_idx + recent:
            if 0 <= i < S and i not in seen:
                keep.append(i)
                seen.add(i)
        reserve = len(seen)
        pin_slots = max(0, budget - reserve)

        pin_hits: list[tuple[float, int]] = []
        for i, a in enumerate(abs_list):
            if i in seen or a not in pin_abs:
                continue
            sc = scores[a] if 0 <= a < len(scores) else 0.0
            pin_hits.append((sc, i))
        pin_hits.sort(key=lambda t: (-t[0], t[1]))
        n_pin_kept = 0
        for _, i in pin_hits:
            if n_pin_kept >= pin_slots:
                break
            if i not in seen:
                keep.append(i)
                seen.add(i)
                n_pin_kept += 1
        last_n_novelty = n_pin_kept

        # Fill by attention rank (query-aware) or sequential
        if attn_rank is not None and len(attn_rank) >= S:
            order = sorted(range(S), key=lambda i: -attn_rank[i])
        else:
            order = list(range(S))
        for i in order:
            if len(keep) >= budget:
                break
            if i not in seen:
                keep.append(i)
                seen.add(i)
        keep = sorted(keep[:budget])
        past_kv = compress_keep_indices(past_kv, keep)
        abs_list = [abs_list[i] for i in keep]
        still = sum(1 for a in abs_list if a in pin_abs)
        last_recall_proxy = still / max(len(pin_abs), 1)
        if sticky:
            sticky_pins.intersection_update(abs_list)
        return past_kv, abs_list

    def _attn_rank_vector(past_kv, end: int) -> list[float] | None:
        """Per-cache-index attention score for query window (last w tokens)."""
        if __package__:
            from .scorer_valley import (
                aggregate_prefix_vote,
                obs_attentions_on_past,
            )
            from .snapkv import full_layer_h_kv
        else:
            from scorer_valley import (
                aggregate_prefix_vote,
                obs_attentions_on_past,
            )
            from snapkv import full_layer_h_kv

        T = end
        S = cache_seq_len(past_kv)
        w = min(window_size, T - 1, S - 1)
        if w < 4:
            return None
        obs_ids = input_ids[:, end - w : end]
        try:
            attns, w, prefix_len = obs_attentions_on_past(
                model, obs_ids, past_kv, abs_obs_start=end - w
            )
            if attns is None or all(a is None for a in attns):
                return None
            h_kv = full_layer_h_kv(past_kv)
            score = aggregate_prefix_vote(
                attns,
                h_kv=h_kv,
                prefix_len=prefix_len,
                window_size=w,
                score_layers=score_layers,
            )
            # score is over prefix; pad/truncate to S
            sc = [0.0] * S
            n = min(S, int(score.numel()), prefix_len)
            for i in range(n):
                sc[i] = float(score[i].item())
            # recent tokens already forced; boost them slightly
            for i in range(max(0, S - window_size), S):
                sc[i] = max(sc[i], 1e6)
            return sc
        except Exception:
            return None

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

        abs_pos.extend(range(start, end))
        S_now = cache_seq_len(past)
        if len(abs_pos) > S_now:
            abs_pos = abs_pos[-S_now:]
        peak_cache = max(peak_cache, S_now)

        # Use hold budget during stream; final tighten later
        budget_now = dyn_budget if end < seq_len else final_budget
        if past is not None and S_now > budget_now:
            nov_pins, last_scores = _novelty_set(end)
            attn_pins = _attention_peak_abs(
                model,
                input_ids[:, :end],
                past,
                abs_pos,
                window_size=window_size,
                expand_radius=expand_radius,
                max_peaks=attn_max_peaks,
                score_layers=score_layers,
            )
            last_n_attn_pins = len(attn_pins)
            pin_abs = nov_pins | attn_pins
            # Mid-stream: no full query-aware rank (question may not exist yet)
            # Near end (last 2 chunks): use attention rank fill
            near_end = end >= seq_len - 2 * chunk_size
            attn_rank = _attn_rank_vector(past, end) if near_end else None
            past, abs_pos = _compress(
                past, abs_pos, budget_now, pin_abs, last_scores, attn_rank
            )
            n_compress += 1
            peak_cache = max(peak_cache, cache_seq_len(past))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert past is not None and last_logits is not None
    # Final query-aware tighten to final_budget (question in last window)
    if cache_seq_len(past) > final_budget:
        nov_pins, last_scores = _novelty_set(seq_len)
        attn_pins = _attention_peak_abs(
            model,
            input_ids,
            past,
            abs_pos,
            window_size=window_size,
            expand_radius=expand_radius,
            max_peaks=attn_max_peaks,
            score_layers=score_layers,
        )
        pin_abs = nov_pins | attn_pins
        attn_rank = _attn_rank_vector(past, seq_len)
        past, abs_pos = _compress(
            past, abs_pos, final_budget, pin_abs, last_scores, attn_rank
        )
        n_compress += 1

    stats = {
        "path": "stream_hybrid_pin",
        "stream_budget": hold_budget,
        "final_budget": final_budget,
        "peak_cache": peak_cache,
        "final_cache": cache_seq_len(past),
        "n_compress": n_compress,
        "last_n_novelty_kept": last_n_novelty,
        "last_n_attn_pins": last_n_attn_pins,
        "last_novelty_retention_proxy": round(last_recall_proxy, 4),
        "floor_ratio": floor_ratio,
        "max_capsules": dyn_max_caps,
        "sticky": sticky,
        "n_sticky_pins": len(sticky_pins),
        "hold_factor": hold_factor,
        "hold_budget": hold_budget,
    }
    return past, last_logits, stats


@torch.inference_mode()
def prefill_streaming_query_hold(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    final_budget: int = 512,
    hold_budget: int = 2048,
    chunk_size: int = 512,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    max_capsules: int = 16,
    score_layers: int = 8,
    attn_max_peaks: int = 24,
) -> tuple[Any, torch.Tensor, dict]:
    """
    Query-hold stream: keep a large intermediate cache (hold_budget) with hybrid
    pins so answer spans are less likely to be dropped before the question
    exists; then query-aware hybrid tighten to final_budget.

    Peak cache ≈ hold_budget + chunk (systems cost of better discovery).
    """
    past, logits, stats = prefill_streaming_hybrid_pin(
        model,
        tokenizer,
        input_ids,
        stream_budget=final_budget,
        final_budget=final_budget,
        chunk_size=chunk_size,
        window_size=window_size,
        sinks=sinks,
        expand_radius=expand_radius,
        max_capsules=max_capsules,
        score_layers=score_layers,
        attn_max_peaks=attn_max_peaks,
        hold_budget=hold_budget,
        hold_factor=1.0,
    )
    stats["path"] = "stream_query_hold"
    stats["hold_budget"] = hold_budget
    stats["final_budget"] = final_budget
    return past, logits, stats
