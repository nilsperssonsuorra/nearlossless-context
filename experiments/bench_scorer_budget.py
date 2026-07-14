"""
Non-oracle scorer budget sweep (post H1/H2).

Question: at near-oracle token budgets (~R*=1 priority set size), can attention
scoring recover critical±R* and match full KV needle quality without an oracle?

Arms:
  full          — gold
  oracle_r1     — sinks ∪ crit±R* ∪ recent  (upper bound for any scorer)
  recent        — recent window only
  snapkv        — existing SnapKV (per-layer indices, expand_radius=2)
  shared_r0     — shared max-vote; expand then *re-cap by token score* (drops neighbors)
  shared_r1     — same with expand_radius=1
  shared_r2     — same with expand_radius=2
  seed_r1       — seed-then-complete: top seeds keep atomic ±R* neighborhoods
  seed_valley   — seeds + grow contiguous high-score segment (not fixed R only)
  snap_union_r1 — SnapKV per-layer top-k **union** → atomic ±R* complete (shared cache)

Primary metrics: success (ε=0), span_recall on critical, span_recall on crit±R*,
cache tokens / KV MB. Reports min budget with 100% success per method.

Usage:
  python experiments/bench_scorer_budget.py
  python experiments/bench_scorer_budget.py --budgets 160,168,176,192,256 --R 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_h1_oracle import (  # noqa: E402
    build_index_set,
    compress_keep_indices,
    expand_critical,
    find_all_critical_spans,
    span_recall,
)
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate, prefill_method  # noqa: E402
from kv_select import attention_to_vote, select_indices_uniform  # noqa: E402
from snapkv import (  # noqa: E402
    cache_nbytes,
    cache_seq_len,
    clone_dynamic_cache,
    crop_cache_prefix,
)
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-oracle scorer vs oracle budget sweep")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument(
        "--budgets",
        default="168,192,224,256,320,384,512",
        help="Total KV token budgets (must be > window)",
    )
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1, help="H1′ radius for oracle + shared expand")
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--methods",
        default="full,oracle_r1,snapkv,seed_r1,seed_valley,snap_union_r1",
    )
    p.add_argument(
        "--score-layers",
        type=int,
        default=8,
        help="Use last N layers' attention for shared vote (0=all)",
    )
    return p.parse_args()


@torch.inference_mode()
def full_prefill(model, input_ids: torch.Tensor):
    out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.inference_mode()
def obs_attentions(
    model,
    input_ids: torch.Tensor,
    past_full,
    window_size: int,
):
    """Eager attn scores for last `window_size` tokens over full prefix+window."""
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
    score_layers: int,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Mean vote over selected layers, max over heads → [prefix_len] on device."""
    usable = [a for a in attns if a is not None]
    if not usable:
        raise RuntimeError("no attentions")
    if score_layers > 0:
        usable = usable[-score_layers:]
    votes = []
    for attn in usable:
        # attn: [B, Hq, q, kv]
        v = attention_to_vote(
            attn,
            h_kv=h_kv,
            prefix_len=prefix_len,
            window_size=window_size,
            kernel_size=kernel_size,
            query_power=2.0,
            use_max=False,
        )  # [B, Hkv, prefix]
        votes.append(v)
    stacked = torch.stack(votes, dim=0).mean(dim=0)  # [B, H, P]
    # shared score: max over heads
    score = stacked.max(dim=1).values[0]  # [P]
    return score


def select_shared_keep(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int,
) -> list[int]:
    """
    sinks ∪ top-scored prefix (with optional ±expand) ∪ recent window.
    Total capped at `budget` (recent+sinks forced).
    """
    budget = min(budget, seq_len)
    recent = list(range(max(0, seq_len - window_size), seq_len))
    sink_idx = list(range(min(sinks, prefix_len, seq_len)))
    forced = set(sink_idx) | set(recent)
    if len(forced) >= budget:
        # prefer recent + sinks trimmed
        keep = sorted(forced)
        # drop lowest-priority mid fillers first — keep all recent, then sinks
        if len(keep) > budget:
            keep = sorted(set(recent) | set(sink_idx[: max(0, budget - len(recent))]))
            keep = keep[:budget]
        return sorted(keep)[:budget]

    need = budget - len(forced)
    # candidate scores on prefix excluding forced
    sc = score_prefix.clone()
    for i in sink_idx:
        if i < sc.numel():
            sc[i] = -1e9
    # only prefix positions (not in window; window is forced via recent)
    n_pref = min(prefix_len, sc.numel())
    sc = sc[:n_pref].clone()
    # top-k generously then expand
    k0 = min(n_pref, max(need * (1 + 2 * expand_radius), need + 8))
    top = torch.topk(sc, k=k0).indices.tolist()
    cand: set[int] = set()
    for t in top:
        if expand_radius > 0:
            for d in range(-expand_radius, expand_radius + 1):
                j = int(t) + d
                if 0 <= j < prefix_len:
                    cand.add(j)
        else:
            cand.add(int(t))
    # rank candidates by original score (center), force not in forced
    ranked = sorted(
        (i for i in cand if i not in forced),
        key=lambda i: float(score_prefix[i].item()) if i < score_prefix.numel() else -1e9,
        reverse=True,
    )
    chosen = set(forced)
    for i in ranked:
        if len(chosen) >= budget:
            break
        chosen.add(i)
    # fill if expand starved
    if len(chosen) < budget:
        order = torch.argsort(score_prefix[:n_pref], descending=True).tolist()
        for i in order:
            if i in chosen:
                continue
            chosen.add(int(i))
            if len(chosen) >= budget:
                break
    return sorted(chosen)[:budget]


def select_seed_complete(
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int,
) -> list[int]:
    """
    H1′-aligned selection: pick high-score *seeds*, then keep each seed's
    ±R neighborhood as an atomic unit (never drop neighbors when capping).

    Why: expand-then-topk-by-token-score discards low-score neighbors of peaks
    (e.g. trailing digits), which is exactly the failure mode of bare spans.
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

    width = 2 * expand_radius + 1
    slots = budget - len(forced)
    # max seed count if each seed costs `width` (overlap reduces real cost)
    max_seeds = max(1, min(n_pref, slots))
    # rank all prefix positions as potential seeds
    order = torch.argsort(sc, descending=True).tolist()

    # Greedy: add seed neighborhoods by seed score; skip seeds already covered
    covered: set[int] = set()
    chosen: set[int] = set(forced)
    seed_scores: list[tuple[float, int, set[int]]] = []  # for optional prune

    for s in order:
        s = int(s)
        if s < 0 or s >= n_pref:
            continue
        if s in covered:
            continue
        if float(sc[s].item()) <= -1e8:
            continue
        nbhd: set[int] = set()
        for d in range(-expand_radius, expand_radius + 1):
            j = s + d
            if 0 <= j < n_pref:
                nbhd.add(j)
        # cost = new tokens not already chosen
        new = nbhd - chosen
        if not new and s in chosen:
            covered |= nbhd
            continue
        if len(chosen) + len(new) > budget:
            # try partial: if seed itself not in, skip; else add what fits by score
            if len(chosen) >= budget:
                break
            # add highest-score members of nbhd that fit (prefer seed center first)
            ranked_nb = sorted(
                new,
                key=lambda i: (0 if i == s else 1, -float(score_prefix[i].item())),
            )
            for j in ranked_nb:
                if len(chosen) >= budget:
                    break
                chosen.add(j)
            covered |= nbhd
            if len(chosen) >= budget:
                break
            continue
        chosen |= new
        covered |= nbhd
        seed_scores.append((float(sc[s].item()), s, nbhd))
        if len(chosen) >= budget:
            break
        # stop if we cannot add another full min-neighborhood usefully
        if len(chosen) + 1 > budget:
            break

    # Fill remainder with next top scores (complete ±R if room)
    if len(chosen) < budget:
        for s in order:
            s = int(s)
            if s in chosen or s < 0 or s >= n_pref:
                continue
            if float(sc[s].item()) <= -1e8:
                continue
            nbhd = set()
            for d in range(-expand_radius, expand_radius + 1):
                j = s + d
                if 0 <= j < n_pref:
                    nbhd.add(j)
            new = nbhd - chosen
            if len(chosen) + len(new) <= budget:
                chosen |= new
            elif s not in chosen and len(chosen) < budget:
                chosen.add(s)
            if len(chosen) >= budget:
                break

    # Final cap (should rarely need): drop lowest-score non-forced singles
    if len(chosen) > budget:
        extras = sorted(
            (i for i in chosen if i not in forced),
            key=lambda i: float(score_prefix[i].item()) if i < score_prefix.numel() else -1e9,
        )
        while len(chosen) > budget and extras:
            chosen.discard(extras.pop(0))

    return sorted(chosen)[:budget]


def grow_valley(
    score_prefix: torch.Tensor,
    seed: int,
    prefix_len: int,
    *,
    floor_ratio: float = 0.25,
    max_radius: int = 24,
) -> set[int]:
    """Grow contiguous segment around seed while score stays above thr."""
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
    """Seeds ranked by score; each gets valley-grown segment ∪ fixed ±R*."""
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
            ranked = sorted(
                new,
                key=lambda i: -float(score_prefix[i].item()),
            )
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


def select_from_seed_set(
    seeds: set[int],
    score_prefix: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    budget: int,
    sinks: int,
    window_size: int,
    expand_radius: int,
) -> list[int]:
    """Complete ±R* around provided seeds (e.g. SnapKV union), force sinks+recent."""
    budget = min(budget, seq_len)
    recent = list(range(max(0, seq_len - window_size), seq_len))
    sink_idx = list(range(min(sinks, prefix_len, seq_len)))
    forced = set(sink_idx) | set(recent)
    chosen: set[int] = set(forced)
    # rank seeds by score
    ranked_seeds = sorted(
        (s for s in seeds if 0 <= s < prefix_len),
        key=lambda i: float(score_prefix[i].item()) if i < score_prefix.numel() else -1e9,
        reverse=True,
    )
    for s in ranked_seeds:
        nbhd = set()
        for d in range(-expand_radius, expand_radius + 1):
            j = s + d
            if 0 <= j < prefix_len:
                nbhd.add(j)
        new = nbhd - chosen
        if len(chosen) + len(new) <= budget:
            chosen |= new
        else:
            for j in sorted(new, key=lambda i: -float(score_prefix[min(i, score_prefix.numel()-1)].item())):
                if len(chosen) >= budget:
                    break
                chosen.add(j)
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:
        n_pref = min(prefix_len, int(score_prefix.numel()))
        order = torch.argsort(score_prefix[:n_pref], descending=True).tolist()
        for i in order:
            i = int(i)
            if i in chosen:
                continue
            chosen.add(i)
            if len(chosen) >= budget:
                break
    return sorted(chosen)[:budget]


@torch.inference_mode()
def run_shared_scorer(
    model,
    input_ids: torch.Tensor,
    *,
    budget: int,
    window_size: int,
    sinks: int,
    expand_radius: int,
    score_layers: int,
    mode: str = "shared",
) -> tuple[object, torch.Tensor, list[int]]:
    past, logits = full_prefill(model, input_ids)
    seq_len = int(input_ids.shape[-1])
    if seq_len <= budget:
        return past, logits, list(range(seq_len))

    attns, window_size, prefix_len = obs_attentions(
        model, input_ids, past, window_size
    )
    if attns is None or all(a is None for a in attns):
        # fallback recent
        keep = list(range(max(0, seq_len - budget), seq_len))
        past = compress_keep_indices(past, keep)
        return past, logits, keep

    h_kv = past.layers[0].keys.shape[1]
    score = aggregate_prefix_vote(
        attns,
        h_kv=h_kv,
        prefix_len=prefix_len,
        window_size=window_size,
        score_layers=score_layers,
    )
    if mode == "seed":
        keep = select_seed_complete(
            score,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
        )
    elif mode == "valley":
        keep = select_seed_valley(
            score,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
        )
    elif mode == "snap_union":
        # Per-layer SnapKV-style top-k (no expand), union seeds, then ±R complete
        keep_prefix = max(budget - window_size, sinks + 1)
        seed_set: set[int] = set()
        usable = [a for a in attns if a is not None]
        for attn in usable:
            vote = attention_to_vote(
                attn,
                h_kv=h_kv,
                prefix_len=prefix_len,
                window_size=window_size,
                kernel_size=7,
                query_power=2.0,
                use_max=False,
            )
            idx = select_indices_uniform(
                vote,
                keep_prefix,
                sink_size=sinks,
                expand_radius=0,
            )  # [B,H,K]
            seed_set.update(int(x) for x in idx[0].unique().tolist())
        keep = select_from_seed_set(
            seed_set,
            score,
            seq_len=seq_len,
            prefix_len=prefix_len,
            budget=budget,
            sinks=sinks,
            window_size=window_size,
            expand_radius=expand_radius,
        )
    else:
        keep = select_shared_keep(
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
    return past, logits, keep


def eval_row(
    *,
    method: str,
    success: bool,
    hits: int,
    keep: list[int] | None,
    critical: list[int],
    critical_r: list[int],
    past,
    answer: str,
    budget: int | None,
    depth: float,
    model_id: str,
    ctx_actual: int,
    R: int,
) -> dict:
    recall_c = span_recall(keep, critical) if keep is not None else None
    recall_r = span_recall(keep, critical_r) if keep is not None else None
    return {
        "model": model_id,
        "method": method,
        "budget": budget if budget is not None else -1,
        "depth": depth,
        "R": R,
        "success": success,
        "hits": hits,
        "span_recall_crit": recall_c,
        "span_recall_crit_R": recall_r,
        "keep_count": len(keep) if keep is not None else cache_seq_len(past),
        "cache_tokens": cache_seq_len(past),
        "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
        "ctx_actual": ctx_actual,
        "answer": answer[:160].replace("\n", " "),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Non-oracle scorer budget sweep ===", flush=True)
    print(
        f"Model={args.model} ctx={args.ctx} R*={args.R} window={args.window}",
        flush=True,
    )
    print(f"depths={depths} budgets={budgets}", flush=True)
    print(f"methods={methods}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU after load: {torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )

    rows: list[dict] = []

    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])
        critical = find_all_critical_spans(tokenizer, input_ids)
        critical_r = expand_critical(critical, seq_len, args.R)
        print(
            f"\n=== depth={depth:.2f} seq={seq_len} "
            f"n_crit={len(critical)} n_crit±R={len(critical_r)} ===",
            flush=True,
        )

        # --- full once ---
        if "full" in methods:
            print("  full…", flush=True)
            past, logits = full_prefill(model, input_ids)
            toks = greedy_generate(
                model,
                past,
                logits,
                args.max_new,
                eos_id=tokenizer.eos_token_id,
                next_position=seq_len,
            )
            sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
            row = eval_row(
                method="full",
                success=sc["success"],
                hits=sc["hits"],
                keep=list(range(seq_len)),
                critical=critical,
                critical_r=critical_r,
                past=past,
                answer=sc["answer"],
                budget=None,
                depth=depth,
                model_id=args.model,
                ctx_actual=seq_len,
                R=args.R,
            )
            rows.append(row)
            print(
                f"    success={row['success']} kv_mb={row['kv_mb']}",
                flush=True,
            )
            del past, logits, toks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not sc["success"]:
                print("  full failed — skip depth", flush=True)
                continue

        # oracle size reference
        keep_ora = build_index_set(
            seq_len,
            sinks=args.sink_size,
            recent=args.window,
            critical=critical,
            mode="oracle_ctx",
            span_context=args.R,
        )
        n_ora = len(keep_ora)
        print(f"  oracle_r{args.R} size={n_ora}", flush=True)

        for method in methods:
            if method == "full":
                continue

            if method == "oracle_r1":
                print(f"  oracle_r{args.R}…", flush=True)
                past, logits = full_prefill(model, input_ids)
                past = compress_keep_indices(past, keep_ora)
                toks = greedy_generate(
                    model,
                    past,
                    logits,
                    args.max_new,
                    eos_id=tokenizer.eos_token_id,
                    next_position=seq_len,
                )
                sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
                row = eval_row(
                    method="oracle_r1",
                    success=sc["success"],
                    hits=sc["hits"],
                    keep=keep_ora,
                    critical=critical,
                    critical_r=critical_r,
                    past=past,
                    answer=sc["answer"],
                    budget=n_ora,
                    depth=depth,
                    model_id=args.model,
                    ctx_actual=seq_len,
                    R=args.R,
                )
                rows.append(row)
                print(
                    f"    success={row['success']} recall_c={row['span_recall_crit']:.2f} "
                    f"recall_R={row['span_recall_crit_R']:.2f} n={row['keep_count']} "
                    f"mb={row['kv_mb']}",
                    flush=True,
                )
                del past, logits, toks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for budget in budgets:
                if budget <= args.window:
                    print(f"  {method}@{budget} skip (≤window)", flush=True)
                    continue
                tag = f"{method}@{budget}"
                print(f"  {tag}…", flush=True)
                try:
                    keep: list[int] | None
                    if method == "recent":
                        past, logits = prefill_method(
                            model,
                            input_ids,
                            None,
                            "recent",
                            budget=budget,
                            window=args.window,
                        )
                        keep = list(range(max(0, seq_len - budget), seq_len))
                    elif method == "snapkv":
                        past, logits = prefill_method(
                            model,
                            input_ids,
                            None,
                            "snapkv",
                            budget=budget,
                            window=args.window,
                            kernel=7,
                        )
                        keep = None  # per-head; recall N/A
                    elif method.startswith("shared_r"):
                        er = int(method.split("_r")[-1])
                        past, logits, keep = run_shared_scorer(
                            model,
                            input_ids,
                            budget=budget,
                            window_size=args.window,
                            sinks=args.sink_size,
                            expand_radius=er,
                            score_layers=args.score_layers,
                            mode="shared",
                        )
                    elif method.startswith("seed_r"):
                        er = int(method.split("_r")[-1])
                        past, logits, keep = run_shared_scorer(
                            model,
                            input_ids,
                            budget=budget,
                            window_size=args.window,
                            sinks=args.sink_size,
                            expand_radius=er,
                            score_layers=args.score_layers,
                            mode="seed",
                        )
                    elif method == "seed_valley":
                        past, logits, keep = run_shared_scorer(
                            model,
                            input_ids,
                            budget=budget,
                            window_size=args.window,
                            sinks=args.sink_size,
                            expand_radius=args.R,
                            score_layers=args.score_layers,
                            mode="valley",
                        )
                    elif method == "snap_union_r1":
                        past, logits, keep = run_shared_scorer(
                            model,
                            input_ids,
                            budget=budget,
                            window_size=args.window,
                            sinks=args.sink_size,
                            expand_radius=1,
                            score_layers=args.score_layers,
                            mode="snap_union",
                        )
                    else:
                        raise ValueError(method)

                    toks = greedy_generate(
                        model,
                        past,
                        logits,
                        args.max_new,
                        eos_id=tokenizer.eos_token_id,
                        next_position=seq_len,
                    )
                    sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
                    row = eval_row(
                        method=method,
                        success=sc["success"],
                        hits=sc["hits"],
                        keep=keep,
                        critical=critical,
                        critical_r=critical_r,
                        past=past,
                        answer=sc["answer"],
                        budget=budget,
                        depth=depth,
                        model_id=args.model,
                        ctx_actual=seq_len,
                        R=args.R,
                    )
                    rows.append(row)
                    rc = (
                        f"{row['span_recall_crit']:.2f}"
                        if row["span_recall_crit"] is not None
                        else "n/a"
                    )
                    rr = (
                        f"{row['span_recall_crit_R']:.2f}"
                        if row["span_recall_crit_R"] is not None
                        else "n/a"
                    )
                    print(
                        f"    success={row['success']} recall_c={rc} recall_R={rr} "
                        f"n={row['cache_tokens']} mb={row['kv_mb']} "
                        f"ans={row['answer'][:50]!r}",
                        flush=True,
                    )
                    del past, logits, toks
                except Exception as e:
                    print(f"    ERROR {e}", flush=True)
                    rows.append(
                        {
                            "method": method,
                            "budget": budget,
                            "depth": depth,
                            "error": str(e),
                            "model": args.model,
                        }
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- summary ---
    print("\n=== Summary: success rate by method × budget ===", flush=True)
    method_names = sorted({r["method"] for r in rows if "method" in r and "success" in r})
    budget_vals = sorted({r.get("budget", -1) for r in rows if "success" in r})
    summary: dict = {}
    for m in method_names:
        summary[m] = {}
        for b in budget_vals:
            sub = [
                r
                for r in rows
                if r.get("method") == m
                and r.get("budget", -1) == b
                and "success" in r
            ]
            if not sub:
                continue
            ok = sum(1 for r in sub if r["success"])
            recalls = [
                r["span_recall_crit"]
                for r in sub
                if r.get("span_recall_crit") is not None
            ]
            mean_rec = sum(recalls) / len(recalls) if recalls else None
            summary[m][str(b)] = {
                "ok": ok,
                "n": len(sub),
                "rate": ok / len(sub),
                "mean_span_recall_crit": mean_rec,
            }
            rec_s = f" recall_c={mean_rec:.2f}" if mean_rec is not None else ""
            print(
                f"  {m:12s} @{str(b):>4s}  {100*ok/len(sub):5.0f}% ({ok}/{len(sub)}){rec_s}",
                flush=True,
            )

    # min budget with 100% across depths (for methods that have budgets)
    min_perfect: dict[str, int | None] = {}
    for m in method_names:
        if m in ("full", "oracle_r1"):
            # single budget
            sub = [r for r in rows if r.get("method") == m and "success" in r]
            min_perfect[m] = (
                sub[0].get("budget")
                if sub and all(r["success"] for r in sub)
                else None
            )
            continue
        perfect_budgets = []
        for b in budgets:
            sub = [
                r
                for r in rows
                if r.get("method") == m and r.get("budget") == b and "success" in r
            ]
            if sub and all(r["success"] for r in sub):
                perfect_budgets.append(b)
        min_perfect[m] = min(perfect_budgets) if perfect_budgets else None

    print("\n=== Min budget with 100% success (all depths) ===", flush=True)
    for m, b in min_perfect.items():
        print(f"  {m:12s}  {b}", flush=True)

    # Verdict
    ora_ok = all(
        r["success"]
        for r in rows
        if r.get("method") == "oracle_r1" and "success" in r
    )
    shared_r1_min = min_perfect.get("shared_r1")
    seed_r1_min = min_perfect.get("seed_r1")
    seed_valley_min = min_perfect.get("seed_valley")
    snap_union_min = min_perfect.get("snap_union_r1")
    snap_min = min_perfect.get("snapkv")
    shared_r0_min = min_perfect.get("shared_r0")
    ora_budget = min_perfect.get("oracle_r1")

    verdict = "INCONCLUSIVE"
    reasons = []
    if not ora_ok:
        verdict = "ORACLE_BASELINE_FAIL"
        reasons.append("oracle_r1 not 100% — fix suite/R first")
    else:
        # best non-oracle min perfect budget
        cands = {
            k: v
            for k, v in {
                "snapkv": snap_min,
                "shared_r1": shared_r1_min,
                "seed_r1": seed_r1_min,
                "seed_valley": seed_valley_min,
                "snap_union_r1": snap_union_min,
                "shared_r0": shared_r0_min,
            }.items()
            if v is not None
        }
        if cands:
            best_m = min(cands, key=lambda k: cands[k])
            best_b = cands[best_m]
            if ora_budget is not None and best_b <= ora_budget + 8:
                verdict = "SCORER_MATCHES_ORACLE"
            elif ora_budget is not None and best_b <= int(ora_budget * 1.15) + 1:
                verdict = "SCORER_NEAR_ORACLE"
            else:
                verdict = "SCORER_NEAR_ORACLE"
            reasons.append(
                f"best non-oracle {best_m}@{best_b} "
                f"(oracle≈{ora_budget}; snapkv={snap_min}; "
                f"seed_r1={seed_r1_min}; seed_valley={seed_valley_min}; "
                f"snap_union_r1={snap_union_min}; shared_r1={shared_r1_min})"
            )
            if snap_union_min is not None and snap_min is not None:
                if snap_union_min < snap_min:
                    reasons.append("snap_union_r1 beats vanilla SnapKV (H1′ complete on union)")
                elif snap_union_min == snap_min:
                    reasons.append("snap_union_r1 matches SnapKV min budget")
            if seed_valley_min is not None and seed_r1_min is not None:
                if seed_valley_min < seed_r1_min:
                    reasons.append("valley grow beats fixed ±R* seed complete")
            if best_b is not None and ora_budget is not None and best_b > ora_budget:
                reasons.append(
                    f"scorer tax ≈ {best_b - int(ora_budget)} tokens "
                    f"({best_b}/{ora_budget})"
                )
        else:
            verdict = "SCORER_GAP"
            reasons.append(
                "no non-oracle method hit 100% on all depths in this budget grid"
            )

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"scorer_budget_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "min_perfect_budget": min_perfect,
        "summary": summary,
        "oracle_size_note": "oracle_r1 keep count is per-depth in CSV",
        "hypothesis": (
            "Non-oracle attention scoring + R* expand recovers critical±R* "
            "at near-oracle budgets"
        ),
        "R": args.R,
        "budgets": budgets,
        "depths": depths,
        "csv": str(out),
        "model": args.model,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
