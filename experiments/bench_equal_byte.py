"""
Equal-slot / equal-byte Pareto: needle quality vs budget for SnapKV vs ByteBudget.

Stays at 4k context (workstation-safe).

Usage:
  python experiments/bench_equal_byte.py
  python experiments/bench_equal_byte.py --budgets 256,512,1024,1536 --depth 0.5
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

from bench_needle import (  # noqa: E402
    NEEDLE_KEYS,
    build_needle_prompt,
    score_answer,
)
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate, prefill_method  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--budgets", default="256,512,768,1024,1536")
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--methods", default="full,recent,snapkv,bytebudget")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"Model={args.model} ctx={args.ctx}", flush=True)
    print(f"depths={depths} budgets={budgets}", flush=True)

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

    rows = []
    # full once per depth (no budget)
    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        if am is not None:
            am = am.to(device)
        print(
            f"\n=== depth={depth:.2f} tokens={input_ids.shape[-1]} ===",
            flush=True,
        )

        for method in methods:
            budget_list = [-1] if method == "full" else budgets
            for budget in budget_list:
                if method == "full" and budget != -1:
                    continue
                tag = f"{method}@{'full' if budget < 0 else budget}"
                print(f"  {tag}…", flush=True)
                try:
                    past, logits = prefill_method(
                        model,
                        input_ids,
                        am,
                        method,
                        budget=max(budget, args.window + 16) if budget > 0 else 1024,
                        window=args.window,
                        kernel=7,
                    )
                    # skip compress methods when budget invalid
                    if method != "full" and budget <= args.window:
                        raise ValueError("budget <= window")

                    tok_ids = greedy_generate(
                        model,
                        past,
                        logits,
                        args.max_new,
                        eos_id=tokenizer.eos_token_id,
                        next_position=int(input_ids.shape[-1]),
                    )
                    answer = tokenizer.decode(tok_ids, skip_special_tokens=True)
                    sc = score_answer(answer)
                    ctoks = cache_seq_len(past)
                    cmb = cache_nbytes(past) / (1024**2)
                    logical_mb = cmb
                    if method == "bytebudget" and hasattr(past, "_bytebudget_stats"):
                        st = past._bytebudget_stats  # type: ignore[attr-defined]
                        logical_mb = st.get("logical_nbytes", 0) / (1024**2)

                    row = {
                        "model": args.model,
                        "method": method,
                        "budget": budget,
                        "depth": depth,
                        "ctx_actual": int(input_ids.shape[-1]),
                        "cache_tokens": ctoks,
                        "runtime_cache_mb": round(cmb, 3),
                        "logical_cache_mb": round(logical_mb, 3),
                        "success": sc["success"],
                        "hits": sc["hits"],
                        "answer": sc["answer"][:200].replace("\n", " "),
                    }
                    rows.append(row)
                    print(
                        f"    ok={row['success']} hits={row['hits']} "
                        f"cache_tok={ctoks} run_mb={row['runtime_cache_mb']} "
                        f"log_mb={row['logical_cache_mb']}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"    ERROR {e}", flush=True)
                    rows.append(
                        {
                            "method": method,
                            "budget": budget,
                            "depth": depth,
                            "error": str(e),
                        }
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"equal_byte_{stamp}.csv"
    write_csv(out, rows)

    # Summary table: mid-depth success vs budget
    print("\n=== mid-depth (0.5) success by budget ===", flush=True)
    mid = [r for r in rows if r.get("depth") == 0.5 and "success" in r]
    for method in methods:
        xs = [r for r in mid if r["method"] == method]
        if not xs:
            continue
        parts = []
        for r in xs:
            b = r.get("budget", -1)
            parts.append(f"{b}:{int(r['success'])}")
        print(f"  {method:12s}  " + "  ".join(parts), flush=True)

    print("\n=== overall success rate ===", flush=True)
    for method in methods:
        sub = [r for r in rows if r.get("method") == method and "success" in r]
        if not sub:
            continue
        rate = sum(1 for r in sub if r["success"]) / len(sub)
        print(f"  {method:12s}  {rate*100:.0f}% ({sum(1 for r in sub if r['success'])}/{len(sub)})")

    meta = {
        "csv": str(out),
        "depths": depths,
        "budgets": budgets,
        "methods": methods,
        "ctx": args.ctx,
        "window": args.window,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
