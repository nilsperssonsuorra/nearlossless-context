"""
Compare full KV vs SnapKV-style fixed budget vs recent-window on ONE model.

Metrics per context length:
  - prefill time, decode tok/s
  - peak VRAM
  - compressed cache tokens / bytes
  - simple generation smoke (first tokens)

Usage:
  python experiments/bench_compare.py --ctx 2048,4096,8192
  python experiments/bench_compare.py --ctx 2048,4096,8192 --budget 1024
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

from config import (  # noqa: E402
    DECODE_NEW_TOKENS,
    FILLER_UNIT,
    PRIMARY_MODEL_ID,
    RESULTS_DIR,
    SNAPKV_KERNEL,
    SNAPKV_MAX_CAPACITY,
    SNAPKV_WINDOW,
)
from snapkv import (  # noqa: E402
    cache_nbytes,
    cache_seq_len,
    compress_recent,
    prefill_with_snapkv,
)
from utils import (  # noqa: E402
    build_prompt,
    gpu_mem_mb,
    reset_peak_mem,
    timed_cuda,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", default="2048,4096")
    p.add_argument("--budget", type=int, default=SNAPKV_MAX_CAPACITY)
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--kernel", type=int, default=SNAPKV_KERNEL)
    p.add_argument("--decode-tokens", type=int, default=DECODE_NEW_TOKENS)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--methods",
        default="full,snapkv,recent",
        help="Comma list: full,snapkv,recent",
    )
    return p.parse_args()


@torch.inference_mode()
def greedy_decode(model, input_ids, attention_mask, past, n_new: int, device: str):
    """Decode n_new tokens using past that already covers the full prompt."""
    # First next-token from last prompt position logits requires a forward on last id
    # if past includes full prompt; use last token as step input with past WITHOUT
    # re-including it — HF past after prefill already has all prompt states, so we
    # sample from a forward that only advances:
    # Convention used here: `past` is cache after full prompt prefill (possibly compressed).
    # We feed the last prompt token again only if cache was built excluding it — it was not.
    # So we do: logits from a dummy step using the last token with past trimmed? Simpler path:
    # re-forward last token with past that ends at S-1. That is complex.
    #
    # Practical approach matching HF generate internals for research:
    # Run model on full input_ids once WITHOUT past to get first logits if past is None;
    # if past is not None and cache_seq_len ~ prompt, feed only last token and
    # use past as if it ends before last token — WRONG if past includes last.
    #
    # Correct: after prefill(input_ids) -> past includes all tokens; next input is
    # sampled from prefill logits[:, -1]. We must keep those logits from prefill.
    raise NotImplementedError


@torch.inference_mode()
def run_method(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    method: str,
    budget: int,
    window: int,
    kernel: int,
    n_new: int,
    device: str,
) -> dict:
    reset_peak_mem()
    torch.cuda.empty_cache()
    mem0 = gpu_mem_mb()

    def prefill():
        if method == "full":
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            return out.past_key_values, out.logits[:, -1, :]
        if method == "snapkv":
            return prefill_with_snapkv(
                model,
                input_ids,
                attention_mask,
                window_size=window,
                max_capacity=budget,
                kernel_size=kernel,
            )
        if method == "recent":
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past = compress_recent(out.past_key_values, budget)
            return past, out.logits[:, -1, :]
        raise ValueError(method)

    (past, last_logits), prefill_s = timed_cuda(prefill)
    mem_prefill = gpu_mem_mb()

    # Greedy decode loop (RoPE positions continue from true prompt length)
    next_id = torch.argmax(last_logits, dim=-1, keepdim=True)  # [B,1]
    prompt_len = int(input_ids.shape[-1])

    def decode_loop():
        nonlocal past, next_id
        outs = []
        cur = next_id
        pos = prompt_len
        for _ in range(n_new):
            position_ids = torch.tensor([[pos]], device=cur.device)
            cache_position = torch.tensor([pos], device=cur.device)
            try:
                o = model(
                    input_ids=cur,
                    past_key_values=past,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                )
            except TypeError:
                o = model(
                    input_ids=cur,
                    past_key_values=past,
                    position_ids=position_ids,
                    use_cache=True,
                )
            past = o.past_key_values
            cur = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
            outs.append(cur)
            pos += 1
        return outs

    decoded, decode_s = timed_cuda(decode_loop)
    mem_dec = gpu_mem_mb()

    n_tok = len(decoded)
    return {
        "method": method,
        "ctx_actual": int(input_ids.shape[-1]),
        "cache_tokens": cache_seq_len(past),
        "cache_mb": round(cache_nbytes(past) / (1024**2), 2),
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "decode_tok_s": round(n_tok / max(decode_s, 1e-6), 2),
        "vram_after_prefill_mb": round(mem_prefill["max_allocated_mb"], 1),
        "vram_after_decode_mb": round(mem_dec["max_allocated_mb"], 1),
        "vram_base_mb": round(mem0["allocated_mb"], 1),
        "budget": budget if method != "full" else -1,
        "n_new": n_tok,
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    ctx_list = [int(x) for x in args.ctx.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"Model: {args.model}")
    print(f"Methods: {methods}  budget={args.budget} window={args.window}")
    print("Loading…")

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
    for L in ctx_list:
        prompt = build_prompt(
            tokenizer,
            L,
            FILLER_UNIT,
            "State the main idea of the context in a few words.",
        )
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        if am is not None:
            am = am.to(device)
        print(f"\n=== ctx {L} (actual {input_ids.shape[-1]}) ===", flush=True)

        for method in methods:
            print(f"  running {method}…", flush=True)
            try:
                row = run_method(
                    model,
                    tokenizer,
                    input_ids,
                    am,
                    method,
                    args.budget,
                    args.window,
                    args.kernel,
                    args.decode_tokens,
                    device,
                )
                row["model"] = args.model
                row["ctx_target"] = L
                rows.append(row)
                print(
                    f"  {method:7s}  cache={row['cache_tokens']:5d}  "
                    f"prefill={row['prefill_s']:7.3f}s  "
                    f"decode={row['decode_tok_s']:7.2f} tok/s  "
                    f"VRAM={row['vram_after_decode_mb']:8.1f} MB  "
                    f"KV={row['cache_mb']:.1f} MB"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  {method}: OOM")
                rows.append(
                    {
                        "model": args.model,
                        "method": method,
                        "ctx_target": L,
                        "error": "OOM",
                    }
                )
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {method}: ERROR {e}")
                rows.append(
                    {
                        "model": args.model,
                        "method": method,
                        "ctx_target": L,
                        "error": str(e),
                    }
                )
                torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"compare_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "model": args.model,
        "budget": args.budget,
        "window": args.window,
        "methods": methods,
        "ctx": ctx_list,
        "csv": str(out),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
