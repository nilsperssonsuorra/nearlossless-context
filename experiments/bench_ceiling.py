"""
Dense context ceiling map (goal-aligned).

For each length L (full KV, no eviction):
  - peak VRAM
  - prefill time, decode tok/s
  - mid-depth needle success (quality proxy for ε)

Also fits a rough L_max estimate from VRAM vs L (linear).

Default lengths stay ≤4k. Use --allow-long for longer (may lag Windows/WDDM).

Usage:
  python experiments/bench_ceiling.py
  python experiments/bench_ceiling.py --lengths 2048,3072,4096
  python experiments/bench_ceiling.py --lengths 2048,4096,6144 --allow-long
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

from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len  # noqa: E402
from utils import gpu_mem_mb, reset_peak_mem, timed_cuda, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense KV context ceiling map")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument(
        "--lengths",
        default="2048,3072,4096",
        help="Comma-separated context lengths",
    )
    p.add_argument(
        "--allow-long",
        action="store_true",
        help="Allow lengths > 4096 (can thrash desktop on WDDM)",
    )
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--skip-needle",
        action="store_true",
        help="Only measure VRAM/speed (faster)",
    )
    p.add_argument(
        "--vram-limit-mb",
        type=float,
        default=22000,
        help="For L_max estimate: usable VRAM budget (leave headroom)",
    )
    return p.parse_args()


@torch.inference_mode()
def run_length(
    model,
    tokenizer,
    L: int,
    *,
    max_new: int,
    device: str,
    do_needle: bool,
) -> dict:
    reset_peak_mem()
    torch.cuda.empty_cache()

    if do_needle:
        prompt = build_needle_prompt(tokenizer, L, depth=0.5)
    else:
        # synthetic fill only (speed/VRAM)
        from config import FILLER_UNIT
        from utils import build_prompt

        prompt = build_prompt(
            tokenizer,
            L,
            FILLER_UNIT,
            "Reply with OK.",
        )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    # no all-ones mask (sanitize)
    attention_mask = None
    actual = int(input_ids.shape[-1])

    mem0 = gpu_mem_mb()

    def prefill():
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    try:
        out, prefill_s = timed_cuda(prefill)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "ctx_target": L,
            "ctx_actual": actual,
            "status": "OOM_prefill",
            "prefill_s": None,
            "decode_tok_s": None,
            "vram_peak_mb": None,
            "kv_mb": None,
            "cache_tokens": None,
            "needle_success": None,
            "needle_hits": None,
        }

    past = out.past_key_values
    logits = out.logits[:, -1, :]
    mem_pre = gpu_mem_mb()
    kv_mb = cache_nbytes(past) / (1024**2)
    ctoks = cache_seq_len(past)

    def decode():
        return greedy_generate(
            model,
            past,
            logits,
            max_new,
            eos_id=tokenizer.eos_token_id,
            next_position=actual,
        )

    try:
        toks, decode_s = timed_cuda(decode)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "ctx_target": L,
            "ctx_actual": actual,
            "status": "OOM_decode",
            "prefill_s": round(prefill_s, 4),
            "decode_tok_s": None,
            "vram_peak_mb": round(mem_pre["max_allocated_mb"], 1),
            "kv_mb": round(kv_mb, 2),
            "cache_tokens": ctoks,
            "needle_success": None,
            "needle_hits": None,
        }

    mem_dec = gpu_mem_mb()
    n = max(len(toks), 1)
    decode_tps = n / max(decode_s, 1e-6)

    needle_success = None
    needle_hits = None
    answer = None
    if do_needle:
        answer = tokenizer.decode(toks, skip_special_tokens=True)
        sc = score_answer(answer)
        needle_success = sc["success"]
        needle_hits = sc["hits"]

    return {
        "ctx_target": L,
        "ctx_actual": actual,
        "status": "ok",
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "decode_tok_s": round(decode_tps, 2),
        "vram_base_mb": round(mem0["allocated_mb"], 1),
        "vram_peak_mb": round(
            max(mem_pre["max_allocated_mb"], mem_dec["max_allocated_mb"]), 1
        ),
        "kv_mb": round(kv_mb, 2),
        "cache_tokens": ctoks,
        "needle_success": needle_success,
        "needle_hits": needle_hits,
        "answer": (answer or "")[:200].replace("\n", " ") if answer else None,
    }


def estimate_l_max(rows: list[dict], vram_limit_mb: float) -> dict | None:
    """Linear fit peak VRAM ≈ a + b * L using successful rows; solve for L at limit."""
    pts = [
        (r["ctx_actual"], r["vram_peak_mb"])
        for r in rows
        if r.get("status") == "ok" and r.get("vram_peak_mb") is not None
    ]
    if len(pts) < 2:
        return None
    # simple two-point or least squares
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = num / den
    a = my - b * mx
    if b <= 1e-9:
        return {"a_mb": a, "b_mb_per_tok": b, "l_max_est": None, "note": "non_positive_slope"}
    l_max = (vram_limit_mb - a) / b
    return {
        "a_mb": round(a, 2),
        "b_mb_per_tok": round(b, 5),
        "vram_limit_mb": vram_limit_mb,
        "l_max_est": int(max(0, l_max)),
        "fit_points": pts,
    }


def main() -> None:
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not args.allow_long:
        too = [L for L in lengths if L > 4096]
        if too:
            print(
                f"Dropping lengths >4096 {too} (pass --allow-long to keep). "
                "Long full-KV can freeze the desktop under WDDM.",
                flush=True,
            )
            lengths = [L for L in lengths if L <= 4096]
    if not lengths:
        raise SystemExit("No lengths to run.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA required for ceiling map")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Model: {args.model}", flush=True)
    print(f"Lengths: {lengths}  needle={not args.skip_needle}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

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
    for L in lengths:
        print(f"\n=== L={L} ===", flush=True)
        row = run_length(
            model,
            tokenizer,
            L,
            max_new=args.max_new,
            device=device,
            do_needle=not args.skip_needle,
        )
        row["model"] = args.model
        row["dtype"] = args.dtype
        rows.append(row)
        print(
            f"  status={row['status']} actual={row.get('ctx_actual')} "
            f"prefill={row.get('prefill_s')}s decode={row.get('decode_tok_s')} tok/s "
            f"VRAM_peak={row.get('vram_peak_mb')} MB KV={row.get('kv_mb')} MB "
            f"needle={row.get('needle_success')}",
            flush=True,
        )
        torch.cuda.empty_cache()

    est = estimate_l_max(rows, args.vram_limit_mb)
    print("\n=== VRAM fit → L_max estimate ===", flush=True)
    if est:
        print(json.dumps(est, indent=2), flush=True)
    else:
        print("  (need ≥2 successful points)", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"ceiling_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "model": args.model,
        "lengths": lengths,
        "dtype": args.dtype,
        "allow_long": args.allow_long,
        "estimate": est,
        "csv": str(out),
        "gpu": torch.cuda.get_device_name(0),
        "goal": "max L at ε≈0 full-KV quality under VRAM limit",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
