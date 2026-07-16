"""
Paper-grade multi-seed rigor pack (primary model, workstation-safe @4k).

Claims tested with N seeds × depths {0, 0.5, 1.0}:

  A) H1 mechanism: full ok; oracle_r1 ok; anti_oracle fail
  B) Scorer tax: min seed_valley budget with ε=0 vs |oracle_r1|; keep-set overlap
  C) Stream@512: ε=0 rate (systems baseline used for L_ε narrative)

Usage:
  python experiments/bench_paper_rigor.py
  python experiments/bench_paper_rigor.py --seeds 0,1,2,3,4 --ctx 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from decode_utils import greedy_generate  # noqa: E402
from scorer_valley import compress_with_seed_valley, prefill_streaming_valley  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed paper rigor: H1 + scorer tax + stream")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated integer seeds")
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--sinks", type=int, default=8)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--tax-budgets",
        default="155,160,168,176,192,208,224,256,320",
        help="seed_valley budgets to sweep for tax",
    )
    p.add_argument("--stream-budget", type=int, default=512)
    p.add_argument(
        "--skip-stream",
        action="store_true",
        help="Skip stream arm (faster H1+tax only)",
    )
    return p.parse_args()


def jaccard(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def keep_breakdown(
    keep: list[int],
    *,
    oracle: list[int],
    critical: list[int],
    crit_r: list[int],
    seq_len: int,
    sinks: int,
    window: int,
) -> dict:
    ks, ora, crit, cr = set(keep), set(oracle), set(critical), set(crit_r)
    recent = set(range(max(0, seq_len - window), seq_len))
    sink_set = set(range(min(sinks, seq_len)))
    return {
        "n_keep": len(ks),
        "n_oracle": len(ora),
        "jaccard_oracle": round(jaccard(ks, ora), 4),
        "frac_keep_in_oracle": round(len(ks & ora) / max(len(ks), 1), 4),
        "frac_oracle_recovered": round(len(ks & ora) / max(len(ora), 1), 4),
        "crit_recall": round(len(ks & crit) / max(len(crit), 1), 4),
        "crit_r_recall": round(len(ks & cr) / max(len(cr), 1), 4),
        "n_extra_not_oracle": len(ks - ora),
        "n_extra_not_forced": len(ks - ora - sink_set - recent),
        "n_missed_oracle": len(ora - ks),
        "n_missed_crit_r": len(cr - ks),
    }


@torch.inference_mode()
def eval_decode(model, tokenizer, past, logits, seq_len: int, max_new: int) -> dict:
    toks = greedy_generate(
        model,
        past,
        logits,
        max_new,
        eos_id=tokenizer.eos_token_id,
        next_position=seq_len,
    )
    return score_answer(tokenizer.decode(toks, skip_special_tokens=True))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tax_budgets = sorted({int(x) for x in args.tax_budgets.split(",") if x.strip()})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Paper rigor (multi-seed) ===", flush=True)
    print(
        f"model={args.model} ctx={args.ctx} seeds={seeds} depths={depths}",
        flush=True,
    )

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

    rows: list[dict] = []
    h1_cells: list[dict] = []
    tax_cells: list[dict] = []
    stream_cells: list[dict] = []

    for seed in seeds:
        for depth in depths:
            prompt = build_needle_prompt(
                tokenizer, args.ctx, depth, seed=seed
            )
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            seq_len = int(input_ids.shape[-1])
            critical = find_all_critical_spans(tokenizer, input_ids)
            crit_r = expand_critical(critical, seq_len, args.R)
            oracle = build_index_set(
                seq_len,
                sinks=args.sinks,
                recent=args.window,
                critical=critical,
                mode="oracle_ctx",
                span_context=args.R,
            )
            print(
                f"\n--- seed={seed} depth={depth} seq={seq_len} "
                f"n_crit={len(critical)} |oracle|={len(oracle)} ---",
                flush=True,
            )

            # Prefill once for H1 arms + tax start
            past_full, logits = prefill_chunked(model, input_ids, chunk_size=512)

            # --- H1: full ---
            sc = eval_decode(
                model, tokenizer, past_full, logits, seq_len, args.max_new
            )
            row = {
                "block": "h1",
                "arm": "full",
                "seed": seed,
                "depth": depth,
                "success": sc["success"],
                "hits": sc["hits"],
                "cache_tokens": cache_seq_len(past_full),
                "kv_mb": round(cache_nbytes(past_full) / (1024**2), 3),
                "n_oracle": len(oracle),
                "span_recall": 1.0,
                "answer": sc["answer"][:100].replace("\n", " "),
            }
            rows.append(row)
            h1_cells.append(row)
            print(f"  full: ok={sc['success']}", flush=True)

            # --- H1: oracle_r1 ---
            past_o, log_o = prefill_chunked(model, input_ids, chunk_size=512)
            past_o = compress_keep_indices(past_o, oracle)
            sc = eval_decode(model, tokenizer, past_o, log_o, seq_len, args.max_new)
            bd = keep_breakdown(
                oracle,
                oracle=oracle,
                critical=critical,
                crit_r=crit_r,
                seq_len=seq_len,
                sinks=args.sinks,
                window=args.window,
            )
            row = {
                "block": "h1",
                "arm": "oracle_r1",
                "seed": seed,
                "depth": depth,
                "success": sc["success"],
                "hits": sc["hits"],
                "cache_tokens": cache_seq_len(past_o),
                "kv_mb": round(cache_nbytes(past_o) / (1024**2), 3),
                "n_oracle": len(oracle),
                "span_recall": span_recall(oracle, critical),
                **{f"bd_{k}": v for k, v in bd.items()},
                "answer": sc["answer"][:100].replace("\n", " "),
            }
            rows.append(row)
            h1_cells.append(row)
            print(
                f"  oracle_r1: ok={sc['success']} n={len(oracle)}",
                flush=True,
            )
            del past_o
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- H1: anti_oracle ---
            anti = build_index_set(
                seq_len,
                sinks=args.sinks,
                recent=args.window,
                critical=critical,
                mode="anti_oracle",
                span_context=args.R,
            )
            past_a, log_a = prefill_chunked(model, input_ids, chunk_size=512)
            past_a = compress_keep_indices(past_a, anti)
            sc = eval_decode(model, tokenizer, past_a, log_a, seq_len, args.max_new)
            row = {
                "block": "h1",
                "arm": "anti_oracle",
                "seed": seed,
                "depth": depth,
                "success": sc["success"],
                "hits": sc["hits"],
                "cache_tokens": cache_seq_len(past_a),
                "kv_mb": round(cache_nbytes(past_a) / (1024**2), 3),
                "n_oracle": len(oracle),
                "span_recall": span_recall(anti, critical),
                "answer": sc["answer"][:100].replace("\n", " "),
            }
            rows.append(row)
            h1_cells.append(row)
            print(
                f"  anti_oracle: ok={sc['success']} recall={row['span_recall']}",
                flush=True,
            )
            del past_a
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- Scorer tax sweep ---
            min_ok_budget = None
            for b in tax_budgets:
                if b <= args.window:
                    continue
                past_t, log_t = prefill_chunked(model, input_ids, chunk_size=512)
                try:
                    past_t, keep = compress_with_seed_valley(
                        model,
                        input_ids,
                        past_t,
                        budget=b,
                        window_size=args.window,
                        sinks=args.sinks,
                        expand_radius=args.R,
                    )
                except Exception as e:
                    rows.append(
                        {
                            "block": "tax",
                            "arm": f"seed_valley@{b}",
                            "seed": seed,
                            "depth": depth,
                            "success": False,
                            "status": f"ERR:{type(e).__name__}",
                            "answer": str(e)[:120],
                        }
                    )
                    del past_t
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                sc = eval_decode(
                    model, tokenizer, past_t, log_t, seq_len, args.max_new
                )
                bd = keep_breakdown(
                    keep,
                    oracle=oracle,
                    critical=critical,
                    crit_r=crit_r,
                    seq_len=seq_len,
                    sinks=args.sinks,
                    window=args.window,
                )
                tax = round(b / max(len(oracle), 1), 4)
                row = {
                    "block": "tax",
                    "arm": f"seed_valley@{b}",
                    "seed": seed,
                    "depth": depth,
                    "budget": b,
                    "success": sc["success"],
                    "hits": sc["hits"],
                    "cache_tokens": cache_seq_len(past_t),
                    "kv_mb": round(cache_nbytes(past_t) / (1024**2), 3),
                    "n_oracle": len(oracle),
                    "tax_vs_oracle": tax,
                    "span_recall": span_recall(keep, critical),
                    **{f"bd_{k}": v for k, v in bd.items()},
                    "answer": sc["answer"][:100].replace("\n", " "),
                }
                rows.append(row)
                tax_cells.append(row)
                if sc["success"] and min_ok_budget is None:
                    min_ok_budget = b
                del past_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Early exit once we have success and a couple higher points not needed
                # Keep sweeping all budgets for curves (paper plots).

            print(
                f"  tax: min_ok_budget={min_ok_budget} |oracle|={len(oracle)} "
                f"tax={(min_ok_budget / len(oracle)) if min_ok_budget else None}",
                flush=True,
            )

            # --- Stream ---
            if not args.skip_stream:
                try:
                    past_s, log_s, st = prefill_streaming_valley(
                        model,
                        input_ids,
                        stream_budget=args.stream_budget,
                        final_budget=args.stream_budget,
                        chunk_size=512,
                        window_size=args.window,
                        sinks=args.sinks,
                        expand_radius=args.R,
                    )
                    sc = eval_decode(
                        model, tokenizer, past_s, log_s, seq_len, args.max_new
                    )
                    row = {
                        "block": "stream",
                        "arm": f"stream_{args.stream_budget}",
                        "seed": seed,
                        "depth": depth,
                        "success": sc["success"],
                        "hits": sc["hits"],
                        "cache_tokens": cache_seq_len(past_s),
                        "kv_mb": round(cache_nbytes(past_s) / (1024**2), 3),
                        "n_oracle": len(oracle),
                        "answer": sc["answer"][:100].replace("\n", " "),
                    }
                    rows.append(row)
                    stream_cells.append(row)
                    print(
                        f"  stream@{args.stream_budget}: ok={sc['success']} "
                        f"cache={row['cache_tokens']}",
                        flush=True,
                    )
                    del past_s
                except Exception as e:
                    rows.append(
                        {
                            "block": "stream",
                            "arm": f"stream_{args.stream_budget}",
                            "seed": seed,
                            "depth": depth,
                            "success": False,
                            "status": f"ERR:{type(e).__name__}",
                            "answer": str(e)[:120],
                        }
                    )
                    print(f"  stream ERROR {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del past_full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Aggregate verdicts ----
    def rate(cells: list[dict], arm: str | None = None) -> dict:
        sel = [c for c in cells if arm is None or c.get("arm") == arm]
        if not sel:
            return {"n": 0, "ok": 0, "rate": None}
        ok = sum(1 for c in sel if c.get("success"))
        return {"n": len(sel), "ok": ok, "rate": round(ok / len(sel), 4)}

    h1_full = rate(h1_cells, "full")
    h1_ora = rate(h1_cells, "oracle_r1")
    h1_anti = rate(h1_cells, "anti_oracle")

    # Min tax budget that succeeds on ALL seed×depth
    by_budget: dict[int, list[bool]] = defaultdict(list)
    oracle_sizes: list[int] = []
    for c in tax_cells:
        if "budget" in c:
            by_budget[int(c["budget"])].append(bool(c.get("success")))
        if c.get("n_oracle"):
            oracle_sizes.append(int(c["n_oracle"]))
    min_global = None
    for b in sorted(by_budget):
        if by_budget[b] and all(by_budget[b]):
            min_global = b
            break
    # Per cell min ok budget
    per_cell_min: dict[tuple, int | None] = {}
    for c in tax_cells:
        key = (c["seed"], c["depth"])
        if c.get("success"):
            b = int(c["budget"])
            if key not in per_cell_min or per_cell_min[key] is None or b < per_cell_min[key]:
                per_cell_min[key] = b
        elif key not in per_cell_min:
            per_cell_min[key] = None

    ok_mins = [v for v in per_cell_min.values() if v is not None]
    mean_oracle = (
        sum(oracle_sizes) / len(oracle_sizes) if oracle_sizes else float("nan")
    )
    mean_min = sum(ok_mins) / len(ok_mins) if ok_mins else None
    mean_tax = (mean_min / mean_oracle) if mean_min and mean_oracle else None

    # Overlap at budget 176 if present
    at_176 = [c for c in tax_cells if c.get("budget") == 176]
    mean_jacc_176 = (
        sum(float(c.get("bd_jaccard_oracle") or 0) for c in at_176) / len(at_176)
        if at_176
        else None
    )
    mean_extra_176 = (
        sum(int(c.get("bd_n_extra_not_oracle") or 0) for c in at_176) / len(at_176)
        if at_176
        else None
    )

    stream_rate = rate(stream_cells)

    h1_supported = (
        h1_full["rate"] == 1.0
        and h1_ora["rate"] == 1.0
        and h1_anti["rate"] == 0.0
    )
    if h1_supported and min_global is not None and (
        stream_rate["rate"] is None or stream_rate["rate"] >= 0.9
    ):
        verdict = "PAPER_RIGOR_OK"
    elif h1_supported:
        verdict = "PAPER_H1_OK_SYSTEMS_PARTIAL"
    else:
        verdict = "PAPER_RIGOR_WEAK"

    reasons = [
        f"H1 full={h1_full} oracle_r1={h1_ora} anti={h1_anti}",
        f"scorer min_budget global_all={min_global}; "
        f"mean_min_ok={mean_min}; mean_|oracle|={round(mean_oracle, 2)}; "
        f"mean_tax={round(mean_tax, 4) if mean_tax else None}",
        f"at@176: mean jaccard(oracle)={mean_jacc_176} "
        f"mean extra_not_oracle={mean_extra_176}",
        f"stream@{args.stream_budget}={stream_rate}",
    ]

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"paper_rigor_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "model": args.model,
        "ctx": args.ctx,
        "seeds": seeds,
        "depths": depths,
        "R": args.R,
        "h1": {"full": h1_full, "oracle_r1": h1_ora, "anti_oracle": h1_anti},
        "scorer_tax": {
            "min_budget_all_cells": min_global,
            "mean_min_ok_budget": mean_min,
            "mean_oracle_size": mean_oracle,
            "mean_tax_ratio": mean_tax,
            "at_176_mean_jaccard_oracle": mean_jacc_176,
            "at_176_mean_extra_not_oracle": mean_extra_176,
            "per_seed_depth_min_ok": {
                f"s{s}_d{d}": v for (s, d), v in sorted(per_cell_min.items())
            },
        },
        "stream": stream_rate,
        "csv": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
