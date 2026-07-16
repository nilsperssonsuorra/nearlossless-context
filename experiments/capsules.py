"""
Fact capsules — atomic memory objects for near-lossless KV compression.

Motivation (H1′): near-lossless retrieval needs critical tokens *plus* a local
neighborhood. Token-level top-k / partial valley fills can split that neighborhood
and corrupt facts. A capsule is an indivisible [lo, hi] span: keep all or drop all.

Query-unknown regime: during streaming prefill the final question is not yet
known. We discover candidate capsules from mid-stream attention scores and pack
as many whole capsules as the budget allows (plus sinks + recent window).

This is the design center for "something new": compress *objects*, not ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class FactCapsule:
    """Contiguous keep/drop unit in *current cache index space*."""

    lo: int
    hi: int  # inclusive
    score: float
    center: int
    source: str = "score"  # score | oracle | forced

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"empty capsule {self.lo}>{self.hi}")

    @property
    def width(self) -> int:
        return int(self.hi - self.lo + 1)

    def indices(self) -> list[int]:
        return list(range(self.lo, self.hi + 1))

    def overlaps(self, other: FactCapsule) -> bool:
        return not (self.hi < other.lo or other.hi < self.lo)

    def merge(self, other: FactCapsule) -> FactCapsule:
        return FactCapsule(
            lo=min(self.lo, other.lo),
            hi=max(self.hi, other.hi),
            score=max(self.score, other.score),
            center=self.center if self.score >= other.score else other.center,
            source=self.source if self.source == other.source else "merged",
        )


@dataclass
class CapsulePackResult:
    keep: list[int]
    capsules_kept: list[FactCapsule]
    capsules_dropped: list[FactCapsule]
    forced: list[int]
    budget: int
    n_tokens_capsules: int
    overflow: bool = False  # True if a capsule was too wide to ever fit
    notes: list[str] = field(default_factory=list)


def grow_capsule(
    score_prefix: torch.Tensor,
    center: int,
    prefix_len: int,
    *,
    expand_radius: int = 1,
    floor_ratio: float = 0.25,
    max_radius: int = 24,
) -> FactCapsule:
    """Valley grow from center, then force ±expand_radius (H1′ atomic radius)."""
    center = int(center)
    n = min(int(prefix_len), int(score_prefix.numel()))
    center = max(0, min(center, n - 1))
    s = float(score_prefix[center].item())
    thr = max(s * floor_ratio, 0.0)
    lo = hi = center
    while lo > 0 and center - lo < max_radius:
        if float(score_prefix[lo - 1].item()) < thr:
            break
        lo -= 1
    while hi + 1 < n and hi - center < max_radius:
        if float(score_prefix[hi + 1].item()) < thr:
            break
        hi += 1
    lo = max(0, lo - int(expand_radius))
    hi = min(n - 1, hi + int(expand_radius))
    return FactCapsule(lo=lo, hi=hi, score=s, center=center, source="score")


def discover_capsules(
    score_prefix: torch.Tensor,
    *,
    prefix_len: int,
    min_sep: int = 64,
    max_capsules: int = 16,
    expand_radius: int = 1,
    floor_ratio: float = 0.25,
    max_radius: int = 24,
    sink_ignore: int = 8,
) -> list[FactCapsule]:
    """
    Query-unknown capsule discovery: peaks in prefix scores → atomic valleys.

    Peaks are local maxima above floor_ratio * global max, separated by min_sep.
    """
    n = min(int(prefix_len), int(score_prefix.numel()))
    if n <= 1:
        return []
    sc = score_prefix[:n].float()
    gmax = float(sc[sink_ignore:n].max().item()) if n > sink_ignore else float(sc.max().item())
    if gmax <= 0:
        return []
    thr = gmax * floor_ratio
    peaks: list[tuple[float, int]] = []
    for i in range(max(sink_ignore, 1), n):
        v = float(sc[i].item())
        if v < thr:
            continue
        left = float(sc[i - 1].item()) if i > 0 else -1e30
        right = float(sc[i + 1].item()) if i + 1 < n else -1e30
        if v >= left and v > right:
            peaks.append((v, i))
    peaks.sort(reverse=True)

    chosen_centers: list[int] = []
    raw: list[FactCapsule] = []
    for v, i in peaks:
        if any(abs(i - c) < min_sep for c in chosen_centers):
            continue
        chosen_centers.append(i)
        raw.append(
            grow_capsule(
                sc,
                i,
                n,
                expand_radius=expand_radius,
                floor_ratio=floor_ratio,
                max_radius=max_radius,
            )
        )
        if len(raw) >= max_capsules:
            break

    # Merge overlapping / adjacent capsules (atomic super-capsules)
    if not raw:
        return []
    raw.sort(key=lambda c: c.lo)
    merged: list[FactCapsule] = [raw[0]]
    for c in raw[1:]:
        prev = merged[-1]
        if c.lo <= prev.hi + 1:
            merged[-1] = prev.merge(c)
        else:
            merged.append(c)
    # Re-sort by score for packing preference
    merged.sort(key=lambda c: -c.score)
    return merged


def oracle_capsules_from_critical(
    critical: list[int],
    *,
    seq_len: int,
    expand_radius: int = 1,
    prefix_len: int | None = None,
) -> list[FactCapsule]:
    """Build oracle capsules = connected components of critical±R (H1′)."""
    if not critical:
        return []
    pl = prefix_len if prefix_len is not None else seq_len
    marked = [False] * seq_len
    for i in critical:
        for d in range(-expand_radius, expand_radius + 1):
            j = i + d
            if 0 <= j < pl:
                marked[j] = True
    caps: list[FactCapsule] = []
    i = 0
    while i < pl:
        if not marked[i]:
            i += 1
            continue
        lo = i
        while i + 1 < pl and marked[i + 1]:
            i += 1
        hi = i
        center = (lo + hi) // 2
        # prefer a true critical as center if present
        for c in critical:
            if lo <= c <= hi:
                center = c
                break
        caps.append(
            FactCapsule(
                lo=lo, hi=hi, score=1e6, center=center, source="oracle"
            )
        )
        i += 1
    return caps


def pack_capsules(
    capsules: list[FactCapsule],
    *,
    seq_len: int,
    budget: int,
    sinks: int = 8,
    window_size: int = 128,
    score_prefix: torch.Tensor | None = None,
    prefix_len: int | None = None,
    fill_remainder: bool = True,
) -> CapsulePackResult:
    """
    Pack whole capsules under budget with forced sinks + recent.

    Phase 1: never partially fill a discovered capsule (atomic).
    Phase 2: if spare slots remain and score_prefix is given, fill by top
    token scores (does not split already-kept capsules; adds new tokens only).

    Pure atomic-only (fill_remainder=False) under-uses budget when few capsules
    are discovered mid-stream — measured failure mode of v0.
    """
    budget = min(int(budget), int(seq_len))
    recent = list(range(max(0, seq_len - window_size), seq_len))
    sink_idx = list(range(min(sinks, seq_len)))
    forced = sorted(set(sink_idx) | set(recent))
    chosen: set[int] = set(forced)
    notes: list[str] = []
    if len(chosen) >= budget:
        keep = sorted(chosen)[:budget]
        return CapsulePackResult(
            keep=keep,
            capsules_kept=[],
            capsules_dropped=list(capsules),
            forced=forced,
            budget=budget,
            n_tokens_capsules=0,
            notes=["forced sinks+recent already fill budget"],
        )

    kept_c: list[FactCapsule] = []
    dropped_c: list[FactCapsule] = []
    overflow = False

    # Prefer higher score; stable tie-break by center
    ordered = sorted(capsules, key=lambda c: (-c.score, c.center))
    for cap in ordered:
        new = [i for i in cap.indices() if 0 <= i < seq_len and i not in chosen]
        if not new:
            kept_c.append(cap)
            continue
        if len(cap.indices()) > budget:
            overflow = True
            dropped_c.append(cap)
            notes.append(f"overflow capsule width={cap.width} center={cap.center}")
            continue
        if len(chosen) + len(new) <= budget:
            chosen.update(new)
            kept_c.append(cap)
        else:
            dropped_c.append(cap)

    spare = budget - len(chosen)
    n_fill = 0
    if fill_remainder and spare > 0 and score_prefix is not None:
        pl = min(
            int(prefix_len if prefix_len is not None else score_prefix.numel()),
            int(score_prefix.numel()),
            seq_len,
        )
        # rank prefix tokens by score; skip already chosen
        sc = score_prefix[:pl].float()
        order = torch.argsort(sc, descending=True).tolist()
        for i in order:
            i = int(i)
            if i in chosen:
                continue
            chosen.add(i)
            n_fill += 1
            if len(chosen) >= budget:
                break
        notes.append(f"fill_remainder={n_fill} after atomic pack")
    elif spare > 0:
        notes.append(f"spare_slots={spare} (atomic-only; no fill)")

    keep = sorted(chosen)[:budget]
    n_cap_tok = len(set().union(*[set(c.indices()) for c in kept_c])) if kept_c else 0
    return CapsulePackResult(
        keep=keep,
        capsules_kept=kept_c,
        capsules_dropped=dropped_c,
        forced=forced,
        budget=budget,
        n_tokens_capsules=n_cap_tok,
        overflow=overflow,
        notes=notes,
    )


def select_keep_capsules(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int = 8,
    window_size: int = 128,
    expand_radius: int = 1,
    min_sep: int = 48,
    max_capsules: int = 24,
    floor_ratio: float = 0.12,
    fill_remainder: bool = True,
) -> tuple[list[int], CapsulePackResult]:
    """Discover + pack capsules → keep index list."""
    caps = discover_capsules(
        score_prefix,
        prefix_len=prefix_len,
        min_sep=min_sep,
        max_capsules=max_capsules,
        expand_radius=expand_radius,
        floor_ratio=floor_ratio,
    )
    pack = pack_capsules(
        caps,
        seq_len=seq_len,
        budget=budget,
        sinks=sinks,
        window_size=window_size,
        score_prefix=score_prefix,
        prefix_len=prefix_len,
        fill_remainder=fill_remainder,
    )
    return pack.keep, pack

def _merge_abs_registry(
    registry: list[FactCapsule],
    new_caps: list[FactCapsule],
    *,
    min_sep: int = 48,
) -> list[FactCapsule]:
    reg = list(registry)
    for c in new_caps:
        merged = False
        for i, r in enumerate(reg):
            if c.overlaps(r) or abs(c.center - r.center) < min_sep:
                if c.overlaps(r):
                    reg[i] = r.merge(c)
                else:
                    reg[i] = c if c.score >= r.score else r
                merged = True
                break
        if not merged:
            reg.append(c)
    reg.sort(key=lambda x: -x.score)
    return reg


def _abs_caps_to_cache(abs_caps: list[FactCapsule], abs_pos: list[int]) -> list[FactCapsule]:
    pos_to_idx = {p: i for i, p in enumerate(abs_pos)}
    out: list[FactCapsule] = []
    for c in abs_caps:
        idxs = [pos_to_idx[p] for p in range(c.lo, c.hi + 1) if p in pos_to_idx]
        if not idxs:
            continue
        lo, hi = min(idxs), max(idxs)
        if len(idxs) / max(c.width, 1) < 0.34:
            continue
        center = pos_to_idx.get(c.center, (lo + hi) // 2)
        out.append(
            FactCapsule(lo=lo, hi=hi, score=c.score, center=int(center), source="sticky")
        )
    return out


@torch.inference_mode()
def compress_past_capsules(
    model,
    input_ids_so_far: torch.Tensor,
    past,
    *,
    budget: int,
    window_size: int = 128,
    sinks: int = 8,
    expand_radius: int = 1,
    score_layers: int = 8,
    min_sep: int = 64,
    max_capsules: int = 16,
    abs_pos: list[int] | None = None,
    abs_registry: list[FactCapsule] | None = None,
) -> tuple[Any, list[int], CapsulePackResult, list[int] | None, list[FactCapsule]]:
    from bench_h1_oracle import compress_keep_indices
    from scorer_valley import aggregate_prefix_vote, obs_attentions_on_past
    from snapkv import cache_seq_len, full_layer_h_kv

    T = int(input_ids_so_far.shape[-1])
    S = cache_seq_len(past)
    registry = list(abs_registry or [])

    if S <= budget:
        keep = list(range(S))
        pack = CapsulePackResult(
            keep=keep,
            capsules_kept=[],
            capsules_dropped=[],
            forced=keep,
            budget=budget,
            n_tokens_capsules=0,
            notes=["under budget"],
        )
        return past, keep, pack, abs_pos, registry

    window_size = min(window_size, T - 1, S - 1)
    if window_size < 1:
        keep = list(range(S))
        pack = CapsulePackResult(
            keep=keep,
            capsules_kept=[],
            capsules_dropped=[],
            forced=keep,
            budget=budget,
            n_tokens_capsules=0,
        )
        return past, keep, pack, abs_pos, registry

    obs_ids = input_ids_so_far[:, -window_size:]
    abs_obs_start = T - window_size
    attns, window_size, prefix_len = obs_attentions_on_past(
        model, obs_ids, past, abs_obs_start=abs_obs_start
    )
    if attns is None or all(a is None for a in attns):
        keep = list(range(max(0, S - budget), S))
        past = compress_keep_indices(past, keep)
        if abs_pos is not None:
            abs_pos = [abs_pos[i] for i in keep]
        pack = CapsulePackResult(
            keep=keep,
            capsules_kept=[],
            capsules_dropped=[],
            forced=keep,
            budget=budget,
            n_tokens_capsules=0,
            notes=["no attentions"],
        )
        return past, keep, pack, abs_pos, registry

    h_kv = full_layer_h_kv(past)
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    new_cache_caps = discover_capsules(
        score,
        prefix_len=prefix_len,
        min_sep=min_sep,
        max_capsules=max_capsules,
        expand_radius=expand_radius,
        floor_ratio=0.12,
    )

    if abs_pos is not None and len(abs_pos) == S:
        new_abs = []
        for c in new_cache_caps:
            alo = abs_pos[c.lo]
            ahi = abs_pos[min(c.hi, S - 1)]
            acenter = abs_pos[min(max(c.center, 0), S - 1)]
            new_abs.append(
                FactCapsule(
                    lo=min(alo, ahi),
                    hi=max(alo, ahi),
                    score=c.score,
                    center=acenter,
                    source="score",
                )
            )
        registry = _merge_abs_registry(registry, new_abs, min_sep=min_sep)
        sticky_cache = _abs_caps_to_cache(registry, abs_pos)
        boosted = [
            FactCapsule(
                lo=c.lo,
                hi=c.hi,
                score=c.score * (1.25 if c.source == "sticky" else 1.0),
                center=c.center,
                source=c.source,
            )
            for c in sticky_cache
        ]
        for c in new_cache_caps:
            if not any(c.overlaps(s) for s in boosted):
                boosted.append(c)
        pack = pack_capsules(
            boosted,
            seq_len=S,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            score_prefix=score,
            prefix_len=prefix_len,
            fill_remainder=True,
        )
        pack.notes = list(pack.notes) + [f"sticky_registry={len(registry)}"]
        keep = pack.keep
    else:
        keep, pack = select_keep_capsules(
            score,
            seq_len=S,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
            min_sep=min_sep,
            max_capsules=max_capsules,
            fill_remainder=True,
        )

    past = compress_keep_indices(past, keep)
    if abs_pos is not None:
        abs_pos = [abs_pos[i] for i in keep]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return past, keep, pack, abs_pos, registry


@torch.inference_mode()
def prefill_streaming_capsules(
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
    min_sep: int = 64,
    max_capsules: int = 16,
) -> tuple[Any, torch.Tensor, dict]:
    """Chunked prefill with atomic + sticky absolute capsules (query-unknown)."""
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
    n_caps_kept = 0
    n_caps_dropped = 0
    last_pack: CapsulePackResult | None = None
    abs_pos: list[int] = []
    abs_registry: list[FactCapsule] = []

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
            past, keep, pack, abs_pos, abs_registry = compress_past_capsules(
                model,
                input_ids[:, :end],
                past,
                budget=dyn_budget,
                window_size=window_size,
                sinks=sinks,
                expand_radius=expand_radius,
                score_layers=score_layers,
                min_sep=min_sep,
                max_capsules=max_capsules,
                abs_pos=abs_pos,
                abs_registry=abs_registry,
            )
            n_compress += 1
            n_caps_kept += len(pack.capsules_kept)
            n_caps_dropped += len(pack.capsules_dropped)
            last_pack = pack
            peak_cache = max(peak_cache, cache_seq_len(past))

    assert past is not None and last_logits is not None
    if cache_seq_len(past) > final_budget:
        past, keep, pack, abs_pos, abs_registry = compress_past_capsules(
            model,
            input_ids,
            past,
            budget=final_budget,
            window_size=window_size,
            sinks=sinks,
            expand_radius=expand_radius,
            score_layers=score_layers,
            min_sep=min_sep,
            max_capsules=max_capsules,
            abs_pos=abs_pos,
            abs_registry=abs_registry,
        )
        n_compress += 1
        n_caps_kept += len(pack.capsules_kept)
        n_caps_dropped += len(pack.capsules_dropped)
        last_pack = pack

    stats = {
        "path": "stream_capsules_sticky",
        "stream_budget": dyn_budget,
        "final_budget": final_budget,
        "peak_cache": peak_cache,
        "final_cache": cache_seq_len(past),
        "n_compress": n_compress,
        "n_caps_kept_events": n_caps_kept,
        "n_caps_dropped_events": n_caps_dropped,
        "sticky_registry_size": len(abs_registry),
        "last_pack": {
            "n_kept": len(last_pack.capsules_kept) if last_pack else 0,
            "n_dropped": len(last_pack.capsules_dropped) if last_pack else 0,
            "n_keep_tokens": len(last_pack.keep) if last_pack else cache_seq_len(past),
            "notes": last_pack.notes if last_pack else [],
        },
        "expand_radius": expand_radius,
        "min_sep": min_sep,
    }
    return past, last_logits, stats
