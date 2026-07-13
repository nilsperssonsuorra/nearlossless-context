"""
H1′: minimum local context radius R around critical spans.

For each depth and radius R in {0,2,4,8,16,32}:
  keep = sinks ∪ critical±R ∪ recent_window
  measure exact needle success vs full KV.

Reports R_min(depth) = smallest R with success==True (ε=0).

Usage:
  python experiments/bench_h1_radius.py
  python experiments/bench_h1_radius.py --radii 0,2,4,8,16,32 --depths 0.0,0.5,1.0
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
    find_all_critical_spans,
    span_recall,
)
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1′ radius sweep for local context")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--radii", default="0,1,2,4,8,12,16,24,32")
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of prompt builds per depth (same builder; reserved for future jitter)",
    )
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        print("Capping ctx to 4096.", flush=True)
        args.ctx = 4096

    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== H1′ local-context radius sweep ===", flush=True)
    print(f"Model={args.model} ctx={args.ctx}", flush=True)
    print(f"depths={depths} radii={radii}", flush=True)
    print(
        f"keep = sinks({args.sink_size}) ∪ crit±R ∪ recent({args.recent_window})",
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

    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])
        critical = find_all_critical_spans(tokenizer, input_ids)
        print(
            f"\n=== depth={depth:.2f} seq={seq_len} n_crit={len(critical)} ===",
            flush=True,
        )

        # Full gold once per depth
        out_full = model(input_ids=input_ids, use_cache=True)
        past_full = out_full.past_key_values
        logits_full = out_full.logits[:, -1, :]
        toks_full = greedy_generate(
            model,
            past_full,
            logits_full,
            args.max_new,
            eos_id=tokenizer.eos_token_id,
            next_position=seq_len,
        )
        ans_full = tokenizer.decode(toks_full, skip_special_tokens=True)
        sc_full = score_answer(ans_full)
        print(f"  full success={sc_full['success']}", flush=True)
        rows.append(
            {
                "model": args.model,
                "depth": depth,
                "radius": -1,
                "arm": "full",
                "success": sc_full["success"],
                "hits": sc_full["hits"],
                "span_recall": 1.0,
                "cache_tokens": cache_seq_len(past_full),
                "kv_mb": round(cache_nbytes(past_full) / (1024**2), 3),
                "ctx_actual": seq_len,
                "answer": sc_full["answer"][:160].replace("\n", " "),
            }
        )
        if not sc_full["success"]:
            print("  SUITE INVALID at this depth — skip radii", flush=True)
            continue

        # One full prefill reused: clone per radius via re-prefill (safest)
        r_min = None
        for R in radii:
            print(f"  R={R}…", flush=True)
            out = model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            keep = build_index_set(
                seq_len,
                sinks=args.sink_size,
                recent=args.recent_window,
                critical=critical,
                mode="oracle" if R == 0 else "oracle_ctx",
                span_context=R,
            )
            past = compress_keep_indices(past, keep)
            toks = greedy_generate(
                model,
                past,
                logits,
                args.max_new,
                eos_id=tokenizer.eos_token_id,
                next_position=seq_len,
            )
            ans = tokenizer.decode(toks, skip_special_tokens=True)
            sc = score_answer(ans)
            rec = span_recall(keep, critical)
            row = {
                "model": args.model,
                "depth": depth,
                "radius": R,
                "arm": "oracle_ctx" if R > 0 else "oracle",
                "success": sc["success"],
                "hits": sc["hits"],
                "span_recall": rec,
                "keep_count": len(keep),
                "cache_tokens": cache_seq_len(past),
                "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                "ctx_actual": seq_len,
                "answer": sc["answer"][:160].replace("\n", " "),
            }
            rows.append(row)
            print(
                f"    success={sc['success']} hits={sc['hits']} "
                f"keep={len(keep)} kv_mb={row['kv_mb']} ans={row['answer'][:70]!r}",
                flush=True,
            )
            if sc["success"] and r_min is None:
                r_min = R
            torch.cuda.empty_cache()

        print(f"  R_min(depth={depth}) = {r_min}", flush=True)

    # Summary table
    print("\n=== R_min by depth (first R with success) ===", flush=True)
    r_mins = {}
    for depth in depths:
        sub = [
            r
            for r in rows
            if r.get("depth") == depth and r.get("radius", -1) >= 0 and "success" in r
        ]
        r_min = None
        for r in sorted(sub, key=lambda x: x["radius"]):
            if r["success"]:
                r_min = r["radius"]
                break
        r_mins[str(depth)] = r_min
        print(f"  depth={depth:.2f}  R_min={r_min}", flush=True)

    # Global R_min = max over depths of R_min(depth)
    vals = [v for v in r_mins.values() if v is not None]
    r_star = max(vals) if vals else None
    print(f"\nR* (cover all tested depths) = {r_star}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h1_radius_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "hypothesis": "H1′ minimum local context radius R around critical spans",
        "r_min_by_depth": r_mins,
        "R_star": r_star,
        "radii": radii,
        "depths": depths,
        "ctx": args.ctx,
        "model": args.model,
        "csv": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
