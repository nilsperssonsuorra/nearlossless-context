"""
Shared KV token selection utilities (improved SnapKV-style voting).

Improvements for mid-context needles:
  - Always keep attention sinks (first sink_size tokens)
  - Weight observation queries toward the end of the window (question side)
  - Max-pool clustering + optional neighbor expand
  - Optional max-over-queries instead of sum (less dilution)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pool1d_max(vote: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """vote: [B, H, S]"""
    if kernel_size <= 1:
        return vote
    pad = kernel_size // 2
    b, h, s = vote.shape
    x = vote.reshape(b * h, 1, s)
    x = F.pad(x, (pad, pad), mode="replicate")
    x = F.max_pool1d(x, kernel_size=kernel_size, stride=1)
    return x.reshape(b, h, s)


def attention_to_vote(
    attn: torch.Tensor,
    *,
    h_kv: int,
    prefix_len: int,
    window_size: int,
    kernel_size: int = 7,
    query_power: float = 2.0,
    use_max: bool = False,
) -> torch.Tensor:
    """
    attn: [B, H_q, q_len, kv_len] from obs-window forward (kv_len == prefix+window).
    Returns vote over prefix only: [B, H_kv, prefix_len].
    """
    bsz, h_q, q_len, kv_len = attn.shape
    group = max(h_q // h_kv, 1)
    # Prefer later queries in the obs window (usually the actual question)
    if q_len > 1:
        w = torch.linspace(0.3, 1.0, q_len, device=attn.device, dtype=attn.dtype)
        w = w.pow(query_power)
        w = w / w.sum()
        # [1,1,q,1]
        w = w.view(1, 1, q_len, 1)
        attn_w = attn * w
    else:
        attn_w = attn

    attn_prefix = attn_w[..., :prefix_len]  # [B, Hq, q, prefix]
    if use_max:
        vote = attn_prefix.max(dim=-2).values
    else:
        vote = attn_prefix.sum(dim=-2)

    if vote.shape[1] != h_kv:
        vote = vote.reshape(bsz, h_kv, group, prefix_len).mean(dim=2)

    vote = pool1d_max(vote, kernel_size)
    return vote


def expand_indices(
    indices: torch.Tensor,
    prefix_len: int,
    radius: int,
) -> torch.Tensor:
    """
    indices: [B, H, K] -> expand neighbors within [0, prefix_len), unique per head,
    padded to max width with last valid index.
    """
    if radius <= 0:
        return indices
    b, h, k = indices.shape
    device = indices.device
    expanded = []
    max_w = 0
    rows = []
    for bi in range(b):
        head_rows = []
        for hi in range(h):
            s: set[int] = set()
            for t in indices[bi, hi].tolist():
                for d in range(-radius, radius + 1):
                    j = int(t) + d
                    if 0 <= j < prefix_len:
                        s.add(j)
            arr = sorted(s)
            head_rows.append(arr)
            max_w = max(max_w, len(arr))
        rows.append(head_rows)

    out = torch.zeros(b, h, max_w, device=device, dtype=indices.dtype)
    for bi in range(b):
        for hi in range(h):
            arr = rows[bi][hi]
            if not arr:
                continue
            t = torch.tensor(arr, device=device, dtype=indices.dtype)
            out[bi, hi, : len(arr)] = t
            if len(arr) < max_w:
                out[bi, hi, len(arr) :] = t[-1]
    return out


def select_indices_uniform(
    vote: torch.Tensor,
    keep: int,
    *,
    sink_size: int = 8,
    expand_radius: int = 0,
) -> torch.Tensor:
    """
    vote: [B, H, prefix_len]
    Always include sinks [0, sink_size), then top-(keep - sink) from non-sink.
    Returns indices [B, H, keep'] sorted (keep' may grow after expand then re-capped).
    """
    b, h, prefix_len = vote.shape
    keep = min(keep, prefix_len)
    sink_size = min(sink_size, keep, prefix_len)

    # Mask sinks out of topk competition so we don't double-count
    vote2 = vote.clone()
    if sink_size > 0:
        vote2[:, :, :sink_size] = -1e9

    rest = keep - sink_size
    if rest <= 0:
        idx = (
            torch.arange(keep, device=vote.device)
            .view(1, 1, -1)
            .expand(b, h, -1)
            .contiguous()
        )
        return idx

    top = torch.topk(vote2, k=min(rest, prefix_len - sink_size), dim=-1).indices
    if sink_size > 0:
        sinks = (
            torch.arange(sink_size, device=vote.device)
            .view(1, 1, -1)
            .expand(b, h, -1)
        )
        idx = torch.cat([sinks, top], dim=-1)
    else:
        idx = top

    idx, _ = torch.sort(idx, dim=-1)
    if expand_radius > 0:
        idx = expand_indices(idx, prefix_len, expand_radius)
        # Re-cap to `keep` by taking unique already sorted and truncating
        # expand may exceed keep — take first `keep` after sort (includes sinks)
        if idx.shape[-1] > keep:
            # Prefer keeping sinks + highest vote among rest
            # Simple truncate after sort keeps earliest positions — better:
            # score each expanded idx by vote and retake top keep with sinks forced
            bsz, hh, m = idx.shape
            # gather votes
            v = torch.gather(vote, 2, idx.clamp(max=prefix_len - 1))
            # force sinks high
            if sink_size > 0:
                is_sink = idx < sink_size
                v = v.masked_fill(is_sink, 1e9)
            top2 = torch.topk(v, k=keep, dim=-1).indices
            idx = torch.gather(idx, 2, top2)
            idx, _ = torch.sort(idx, dim=-1)
    return idx


def select_indices_ada(
    vote: torch.Tensor,
    total_keep: int,
    *,
    sink_size: int = 8,
    alpha: float = 0.35,
    expand_radius: int = 2,
    min_per_head: int | None = None,
) -> torch.Tensor:
    """
    Build a **shared** token set of size `total_keep` (same indices for every KV head).

    Why shared: if each head keeps a different short list and we pad to max_k,
    effective sequence length collapses to max_k + window (often << budget).

    Algorithm:
      1) Ada-style per-head budgets (with a solid min_per_head floor)
      2) Take each head's top-bh positions
      3) Score unique positions by max vote across heads
      4) Force sinks, expand clusters, re-cap to total_keep
      5) Broadcast [B,1,K] -> [B,H,K]
    """
    b, h, prefix_len = vote.shape
    total_keep = min(total_keep, prefix_len)
    sink_size = min(sink_size, total_keep, prefix_len)

    # Default floor: don't starve heads when budget is small (e.g. 512)
    if min_per_head is None:
        # aim ~ total_keep / (2h) but at least 16 when possible
        min_per_head = max(16, total_keep // (2 * h))
    min_per_head = min(min_per_head, total_keep // h if h else total_keep)
    min_per_head = max(min_per_head, min(sink_size, total_keep))

    v0 = vote[0]
    flat = v0.reshape(-1)
    k_g = min(max(total_keep * 2, total_keep), flat.numel())
    top_idx = torch.topk(flat, k=k_g, dim=-1).indices
    head_ids = top_idx // prefix_len
    counts = torch.bincount(head_ids, minlength=h).float()
    uniform = torch.full((h,), total_keep / max(h, 1), device=vote.device)
    blended = alpha * uniform + (1.0 - alpha) * counts
    blended = blended / blended.sum().clamp(min=1e-8) * (total_keep * 1.5)
    # slightly oversample per-head tops so union is rich before re-cap
    budgets = blended.round().long().clamp(min=min_per_head, max=prefix_len)

    # Collect candidate positions + max score across heads
    # score[pos] = max_h vote[h,pos]
    max_vote = vote.max(dim=1).values  # [B, S]
    cand_scores = max_vote.clone()
    # boost positions that appear in many heads' private top lists
    boost = torch.zeros_like(cand_scores)
    for hi in range(h):
        bh = int(budgets[hi].item())
        bh = min(bh, prefix_len)
        top = torch.topk(vote[:, hi, :], k=bh, dim=-1).indices  # [B, bh]
        boost.scatter_add_(1, top, torch.ones_like(top, dtype=boost.dtype))
    cand_scores = cand_scores + 0.05 * boost

    # Force sinks
    if sink_size > 0:
        cand_scores[:, :sink_size] = cand_scores[:, :sink_size] + 1e6

    # Initial top-k
    idx = torch.topk(cand_scores, k=total_keep, dim=-1).indices  # [B, K]
    idx, _ = torch.sort(idx, dim=-1)
    idx = idx.unsqueeze(1)  # [B,1,K]

    if expand_radius > 0:
        idx = expand_indices(idx, prefix_len, expand_radius)
        if idx.shape[-1] > total_keep:
            v = torch.gather(
                cand_scores.unsqueeze(1).expand(-1, 1, -1),
                2,
                idx.clamp(max=prefix_len - 1),
            )
            if sink_size > 0:
                v = v.masked_fill(idx < sink_size, 1e9)
            top2 = torch.topk(v, k=total_keep, dim=-1).indices
            idx = torch.gather(idx, 2, top2)
            idx, _ = torch.sort(idx, dim=-1)

    # Broadcast same selection to all heads
    return idx.expand(b, h, -1).contiguous()
