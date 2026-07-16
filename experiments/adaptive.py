"""
Adaptive R / budget schedule from lab measurements (Qwen3-4B, needle-class).

Not learned online — a compact policy that matches FINDINGS.md thresholds.
Use as default knobs for posthoc/stream compress when task class is known
or estimated (n_entities, L).

Cross-model floors (transfer smoke) are applied via ``model_id`` / family
calibration so Qwen2.5 / Llama-3.2 / Gemma-4 get safer budgets than primary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AdaptivePolicy:
    """Compress policy for seed_valley / stream."""

    R: int
    budget: int
    stream_budget: int
    mode: str = "valley"
    note: str = ""
    family: str = "primary"


def infer_model_family(model_id: str | None) -> str:
    """Map HF id / path → calibration family key."""
    if not model_id:
        return "primary"
    m = str(model_id).lower().replace("\\", "/")
    # Prefer more specific substrings first
    if "gemma-4" in m or "gemma4" in m or "/gemma-4" in m:
        return "gemma4"
    if "qwen2.5" in m or "qwen2_5" in m:
        return "qwen25"
    if "llama-3.2" in m or "llama3.2" in m or "llama-3_2" in m:
        return "llama32"
    if "qwen3" in m:
        return "primary"
    return "primary"


def model_id_from_model(model) -> str:
    """Best-effort HF name from a loaded model."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return ""
    for attr in ("_name_or_path", "name_or_path"):
        v = getattr(cfg, attr, None)
        if v:
            return str(v)
    # Multimodal wrappers sometimes nest text config only — try root
    return str(getattr(cfg, "model_type", "") or "")


def calibrate_policy(
    pol: AdaptivePolicy,
    *,
    model_id: str | None = None,
    L: int = 4096,
    prefer_stream: bool = False,
) -> AdaptivePolicy:
    """
    Raise R/budgets to transfer-measured floors for non-primary families.

    Primary (Qwen3-4B) schedule is left unchanged. Floors from FINDINGS transfer:
      qwen25  — posthoc ≥320; stream ≥768 when L≥8k
      llama32 — posthoc ≥256; stream@512 ok
      gemma4  — posthoc@176 ok; stream novelty@512 multi-seed OK
                (attn/valley still needs ~1024 — see compress_adaptive attn floor)
    """
    fam = infer_model_family(model_id)
    if fam == "primary":
        return replace(pol, family=fam) if pol.family != fam else pol

    R, budget, stream = pol.R, pol.budget, pol.stream_budget
    extra = f"cal={fam}"

    if fam == "qwen25":
        budget = max(budget, 320)
        if L >= 8192:
            stream = max(stream, 768)
        extra += " posthoc≥320" + (" stream≥768@8k+" if L >= 8192 else "")
    elif fam == "llama32":
        budget = max(budget, 256)
        extra += " posthoc≥256"
    elif fam == "gemma4":
        # Hybrid: posthoc seed_valley@176 works after score-pass fix.
        # Novelty stream@512 multi-seed 9/9 @4k (novelty_longL_…181508Z);
        # valley@512 only end-depth (33%) — attn path floors separately.
        if prefer_stream:
            stream = max(stream, 512)
            if L >= 28000:
                stream = max(stream, 2048)
            elif L >= 16000:
                stream = max(stream, 1536)
            extra += " stream≥512 novelty (valley needs ~1024)"
        else:
            budget = max(budget, 176)
            extra += " posthoc≥176"
    else:
        return replace(pol, family=fam)

    note = f"{pol.note} | {extra}".strip(" |")
    return AdaptivePolicy(
        R=R,
        budget=budget,
        stream_budget=stream,
        mode=pol.mode,
        note=note,
        family=fam,
    )


def policy_for(
    *,
    n_entities: int = 1,
    L: int = 4096,
    multi_hop: bool = False,
    prefer_stream: bool = False,
    model_id: str | None = None,
) -> AdaptivePolicy:
    """
    Return R + budgets that empirically hit ε≈0 on our suite.

    n_entities: distinct secrets / critical clusters expected
    L: context length (affects stream budget for single-needle long L)
    multi_hop: two+ linked facts for one answer
    model_id: HF id for cross-model floors (optional; primary if omitted)
    """
    n_entities = max(1, int(n_entities))
    L = int(L)

    # Multi-hop (single answer): scorers worked @512 R=1; keep modest slack
    if multi_hop and n_entities <= 2:
        pol = AdaptivePolicy(
            R=1,
            budget=512,
            stream_budget=512,
            note="two-hop Alice→id→password @4k",
        )
    elif multi_hop and n_entities >= 3:
        pol = AdaptivePolicy(
            R=8,
            budget=768,
            stream_budget=1024,
            note="3-hop / multi-link: prefer larger R + stream slack",
        )
    # Multi-needle recall_all style
    elif n_entities >= 3:
        pol = AdaptivePolicy(
            R=8,
            budget=384,
            stream_budget=1024,
            note="3-needle posthoc R=8@384 / stream R=8@1024",
        )
    elif n_entities == 2:
        # Prefer multi-needle-like R (partial peaks) over tight 2-hop budget
        pol = AdaptivePolicy(
            R=8,
            budget=384,
            stream_budget=1024,
            note="2-entity: use multi-needle R=8 schedule (safer than 2-hop R=1)",
        )
    # Single needle: novelty discovery keeps stream@512 strong through 16k
    # multi-seed (novelty_longL_20260716T180614Z). Attn-only still needs 1536+.
    elif L <= 4096:
        # Posthoc: multi-seed tax → 192. Stream+novelty ~93–100% @512 multi-seed.
        pol = AdaptivePolicy(
            R=1,
            budget=192,
            stream_budget=512 if prefer_stream else 192,
            note="single @4k: posthoc@192 / stream@512 (discovery=novelty)",
        )
    elif L <= 32768:
        # Sticky novelty multi-seed 9/9 @16k/24k/32k stream@512 (FINDINGS).
        pol = AdaptivePolicy(
            R=1,
            budget=192 if L <= 8192 else 256,
            stream_budget=512 if prefer_stream else (192 if L <= 8192 else 256),
            note="single ≤32k: stream@512 sticky-novelty multi-seed 9/9 (peak~1k)",
        )
    elif L <= 40960:
        # 40k sticky multi-seed not re-run; keep modest slack
        pol = AdaptivePolicy(
            R=1,
            budget=256,
            stream_budget=768 if prefer_stream else 256,
            note="single ~40k: stream@768 (32k sticky@512 solid; 40k unmeasured)",
        )
    else:
        # Beyond tested sticky multi-seed envelope
        pol = AdaptivePolicy(
            R=1,
            budget=320,
            stream_budget=1536,
            note="single >40k: stream@1536 slack until sticky multi-seed measured",
        )

    return calibrate_policy(
        pol, model_id=model_id, L=L, prefer_stream=prefer_stream
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
    model_id: str | None = None,
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
        model_id=model_id,
    )
    return pol, n_hat
