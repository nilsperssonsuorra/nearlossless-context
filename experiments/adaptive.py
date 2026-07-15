"""
Adaptive R / budget schedule from lab measurements (Qwen3-4B, needle-class).

Not learned online — a compact policy that matches FINDINGS.md thresholds.
Use as default knobs for posthoc/stream compress when task class is known
or estimated (n_entities, L).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptivePolicy:
    """Compress policy for seed_valley / stream."""

    R: int
    budget: int
    stream_budget: int
    mode: str = "valley"
    note: str = ""


def policy_for(
    *,
    n_entities: int = 1,
    L: int = 4096,
    multi_hop: bool = False,
    prefer_stream: bool = False,
) -> AdaptivePolicy:
    """
    Return R + budgets that empirically hit ε≈0 on our suite.

    n_entities: distinct secrets / critical clusters expected
    L: context length (affects stream budget for single-needle long L)
    multi_hop: two+ linked facts for one answer
    """
    n_entities = max(1, int(n_entities))
    L = int(L)

    # Multi-hop (single answer): scorers worked @512 R=1; keep modest slack
    if multi_hop and n_entities <= 2:
        return AdaptivePolicy(
            R=1,
            budget=512,
            stream_budget=512,
            note="two-hop Alice→id→password @4k",
        )
    if multi_hop and n_entities >= 3:
        return AdaptivePolicy(
            R=8,
            budget=768,
            stream_budget=1024,
            note="3-hop / multi-link: prefer larger R + stream slack",
        )

    # Multi-needle recall_all style
    if n_entities >= 3:
        return AdaptivePolicy(
            R=8,
            budget=384,
            stream_budget=1024,
            note="3-needle posthoc R=8@384 / stream R=8@1024",
        )
    if n_entities == 2:
        # Prefer multi-needle-like R (partial peaks) over tight 2-hop budget
        return AdaptivePolicy(
            R=8,
            budget=384,
            stream_budget=1024,
            note="2-entity: use multi-needle R=8 schedule (safer than 2-hop R=1)",
        )

    # Single needle: scale stream budget with L
    if L <= 4096:
        return AdaptivePolicy(
            R=1,
            budget=176,
            stream_budget=512 if prefer_stream else 176,
            note="single @4k: posthoc@176 / stream@512",
        )
    if L <= 8192:
        return AdaptivePolicy(
            R=1,
            budget=176,
            stream_budget=512,
            note="single @8k stream@512",
        )
    if L <= 12288:
        return AdaptivePolicy(
            R=1,
            budget=256,
            stream_budget=1024,
            note="single ~12k: raise stream (mid fails @512)",
        )
    # 16k–24k class (measured: stream@1536 passes all depths through 24k)
    return AdaptivePolicy(
        R=1,
        budget=256,
        stream_budget=1536,
        note="single @16k–24k stream@1536 (peak cache~2k, VRAM~9GB)",
    )


def estimate_n_entities_from_scores(
    score_prefix,
    *,
    prefix_len: int,
    min_sep: int = 120,
    max_peaks: int = 8,
    floor_ratio: float = 0.12,
    n_bins: int = 16,
    bin_floor: float = 0.05,
    sink_ignore: int = 8,
    topk: int = 40,
) -> int:
    """
    Estimate #critical clusters from attention scores.

    Method: zero sink mass, take top-k score positions, cluster by gap >= min_sep.
    Also count spatial bins with mass >= bin_floor * gmax. Return max of both.
    """
    n = min(prefix_len, int(score_prefix.numel()))
    if n <= 2:
        return 1
    sc = score_prefix[:n].float().clone()
    sc[: min(sink_ignore, n)] = 0.0
    gmax = float(sc.max().item())
    if gmax <= 0:
        return 1

    # --- cluster top-k positions ---
    k = min(topk, n - sink_ignore)
    if k <= 0:
        return 1
    # mask sinks already 0; topk over full then filter
    vals, idxs = sc.topk(k=min(topk, n))
    positions = sorted(int(i) for i, v in zip(idxs.tolist(), vals.tolist()) if v > 0 and int(i) >= sink_ignore)
    clusters: list[list[int]] = []
    for p in positions:
        if not clusters or p - clusters[-1][-1] >= min_sep:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    # keep clusters whose max score is decent
    thr = gmax * floor_ratio
    strong = 0
    for cl in clusters:
        if max(float(sc[i].item()) for i in cl) >= thr:
            strong += 1
    n_top = max(1, min(max_peaks, strong if strong else len(clusters)))

    # --- local maxima ---
    peaks = []
    for i in range(sink_ignore, n):
        v = float(sc[i].item())
        if v < thr:
            continue
        left = float(sc[i - 1].item()) if i > 0 else -1e30
        right = float(sc[i + 1].item()) if i + 1 < n else -1e30
        if v >= left and v > right:
            peaks.append((v, i))
    peaks.sort(reverse=True)
    chosen: list[int] = []
    for _, i in peaks:
        if any(abs(i - c) < min_sep for c in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= max_peaks:
            break
    n_peak = max(1, len(chosen))

    # --- spatial bins ---
    n_bins = max(4, min(n_bins, n // 32 if n >= 64 else 4))
    bin_size = max(1, (n + n_bins - 1) // n_bins)
    bin_hits = 0
    for b in range(n_bins):
        lo = b * bin_size
        hi = min(n, (b + 1) * bin_size)
        if lo >= hi:
            continue
        bmax = float(sc[lo:hi].max().item())
        if bmax >= gmax * bin_floor:
            bin_hits += 1
    n_bin = max(1, min(max_peaks, bin_hits))

    return max(n_peak, n_top, n_bin if n_bin >= 2 else 1)


def policy_from_scores(
    score_prefix,
    *,
    L: int,
    prefix_len: int,
    multi_hop: bool = False,
    prefer_stream: bool = False,
    min_sep: int = 200,
) -> tuple[AdaptivePolicy, int]:
    """Estimate n_entities from scores then return (policy, n_hat)."""
    n_hat = estimate_n_entities_from_scores(
        score_prefix, prefix_len=prefix_len, min_sep=min_sep
    )
    # Do NOT infer multi_hop from peak count — multi-needle needs high R,
    # multi-hop with n_hat=2 was incorrectly mapped to the 2-hop R=1@512 path.
    pol = policy_for(
        n_entities=n_hat,
        L=L,
        multi_hop=multi_hop,
        prefer_stream=prefer_stream,
    )
    return pol, n_hat
