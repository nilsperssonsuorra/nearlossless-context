"""
Measure the long-context tax on one primary model (Qwen3.5-4B).

For each context length L:
  - peak VRAM after prefill
  - TTFT (prefill time)
  - decode tok/s for DECODE_NEW_TOKENS tokens

Usage (from repo root, venv active):
  python experiments/bench_context_tax.py
  python experiments/bench_context_tax.py --ctx 2048,4096
  python experiments/bench_context_tax.py --model Qwen/Qwen3.5-4B --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python experiments/bench_context_tax.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    DECODE_NEW_TOKENS,
    DEFAULT_CTX_LENGTHS,
    FILLER_UNIT,
    PRIMARY_MODEL_ID,
    RESULTS_DIR,
)
from utils import (  # noqa: E402
    build_prompt,
    gpu_mem_mb,
    reset_peak_mem,
    timed_cuda,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-context tax benchmark (single model)")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument(
        "--ctx",
        default=",".join(str(x) for x in DEFAULT_CTX_LENGTHS),
        help="Comma-separated context lengths in tokens",
    )
    p.add_argument("--decode-tokens", type=int, default=DECODE_NEW_TOKENS)
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention implementation (flash_attention_2 needs flash-attn installed)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: results/context_tax_<timestamp>.csv)",
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


@torch.inference_mode()
def run_one(
    model,
    tokenizer,
    ctx_len: int,
    decode_tokens: int,
    device: str,
) -> dict:
    instruction = "Summarize the context in one sentence focusing on the main topic."
    prompt = build_prompt(tokenizer, ctx_len, FILLER_UNIT, instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    actual_len = int(input_ids.shape[-1])
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    reset_peak_mem()
    mem_before = gpu_mem_mb()

    # Prefill = forward that builds KV for the full prompt (first generate step)
    def prefill_and_decode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=decode_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        return out

    # Separate TTFT-ish: one forward for prefill only
    def prefill_only():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    _, ttft_s = timed_cuda(prefill_only)
    mem_after_prefill = gpu_mem_mb()

    # Full generate for end-to-end decode rate (includes prefill — report both)
    reset_peak_mem()
    gen_out, total_s = timed_cuda(prefill_and_decode)
    mem_after_gen = gpu_mem_mb()

    new_tokens = int(gen_out.shape[-1] - actual_len)
    # Approximate decode-only time = total - prefill (rough; good enough for curves)
    decode_s = max(total_s - ttft_s, 1e-6)
    decode_tps = new_tokens / decode_s if new_tokens > 0 else 0.0
    e2e_tps = new_tokens / total_s if new_tokens > 0 else 0.0

    return {
        "ctx_target": ctx_len,
        "ctx_actual": actual_len,
        "decode_new_tokens": new_tokens,
        "ttft_s": round(ttft_s, 4),
        "total_generate_s": round(total_s, 4),
        "decode_approx_s": round(decode_s, 4),
        "decode_tok_s": round(decode_tps, 2),
        "e2e_new_tok_s": round(e2e_tps, 2),
        "vram_before_mb": round(mem_before["allocated_mb"], 1),
        "vram_after_prefill_mb": round(mem_after_prefill["max_allocated_mb"], 1),
        "vram_after_gen_mb": round(mem_after_gen["max_allocated_mb"], 1),
        "vram_reserved_mb": round(mem_after_gen["reserved_mb"], 1),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    ctx_lengths = [int(x.strip()) for x in args.ctx.split(",") if x.strip()]
    dtype = dtype_from_name(args.dtype)

    print(f"Model:   {args.model}")
    print(f"Device:  {args.device}  dtype={args.dtype}  attn={args.attn}")
    print(f"Ctx:     {ctx_lengths}")
    print(f"Decode:  {args.decode_tokens} new tokens")
    print("Loading model…")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = dict(
        trust_remote_code=True,
        dtype=dtype,  # transformers >=4.x (torch_dtype deprecated)
        device_map="auto" if args.device.startswith("cuda") else None,
        attn_implementation=args.attn,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    except Exception as e:
        if args.attn == "flash_attention_2":
            print(f"flash_attention_2 failed ({e}); falling back to sdpa")
            load_kw["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        else:
            raise

    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()

    rows = []
    for L in ctx_lengths:
        print(f"\n=== context target {L} ===")
        try:
            row = run_one(model, tokenizer, L, args.decode_tokens, args.device)
            row["model"] = args.model
            row["dtype"] = args.dtype
            row["attn"] = load_kw.get("attn_implementation", args.attn)
            rows.append(row)
            print(
                f"  actual={row['ctx_actual']}  TTFT={row['ttft_s']}s  "
                f"decode~{row['decode_tok_s']} tok/s  "
                f"VRAM peak gen={row['vram_after_gen_mb']} MB"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at target {L}")
            rows.append(
                {
                    "model": args.model,
                    "dtype": args.dtype,
                    "attn": load_kw.get("attn_implementation", args.attn),
                    "ctx_target": L,
                    "ctx_actual": -1,
                    "decode_new_tokens": 0,
                    "ttft_s": None,
                    "total_generate_s": None,
                    "decode_approx_s": None,
                    "decode_tok_s": None,
                    "e2e_new_tok_s": None,
                    "vram_before_mb": None,
                    "vram_after_prefill_mb": None,
                    "vram_after_gen_mb": None,
                    "vram_reserved_mb": None,
                    "error": "OOM",
                }
            )
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  error: {e}")
            rows.append(
                {
                    "model": args.model,
                    "ctx_target": L,
                    "error": str(e),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else RESULTS_DIR / f"context_tax_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "ctx_lengths": ctx_lengths,
        "decode_tokens": args.decode_tokens,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "csv": str(out),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
