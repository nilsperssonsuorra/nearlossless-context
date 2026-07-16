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
    from bench_h1_oracle import compress_keep_indices
    from snapkv import cache_seq_len

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
