"""
Needle-in-a-haystack smoke test (max 4k tokens — workstation-safe).

Builds a filler haystack, plants a unique secret at a given depth,
asks the model to recall it under full / recent / snapkv / bytebudget.

Usage:
  python experiments/bench_needle.py
  python experiments/bench_needle.py --ctx 4096 --depths 0.0,0.5,1.0 --budget 1024
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    PRIMARY_MODEL_ID,
    RESULTS_DIR,
    SNAPKV_KERNEL,
    SNAPKV_MAX_CAPACITY,
    SNAPKV_WINDOW,
)
from decode_utils import greedy_generate, prefill_method  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len  # noqa: E402
from utils import write_csv  # noqa: E402

NEEDLE = (
    "The secret project code is BLUE-ORBIT-7742 and the archive password is maple-quartz-19."
)
NEEDLE_KEYS = ["BLUE-ORBIT-7742", "maple-quartz-19"]
QUESTION = (
    "What is the secret project code and the archive password mentioned in the context? "
    "Answer with the exact code and password only."
)
FILLER = (
    "Routine status log: sensors nominal, telemetry within range, no anomalies detected. "
    "Maintenance checklist item completed; continue standard operations as scheduled. "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096, help="Target total prompt tokens (≤4096)")
    p.add_argument("--depths", default="0.0,0.5,1.0", help="Needle depth in [0,1]")
    p.add_argument("--budget", type=int, default=SNAPKV_MAX_CAPACITY)
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--kernel", type=int, default=SNAPKV_KERNEL)
    p.add_argument(
        "--methods",
        default="full,recent,snapkv,bytebudget",
    )
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args()


def build_needle_prompt(tokenizer, target_tokens: int, depth: float) -> str:
    """depth=0 → needle near start; 1 → near end (before question)."""
    depth = max(0.0, min(1.0, depth))
    # Grow filler then insert needle; append question at end (obs window)
    body_budget = max(target_tokens - 128, 256)
    chunks: list[str] = []
    while True:
        text_try = "".join(chunks)
        n = len(tokenizer.encode(text_try, add_special_tokens=False))
        if n >= body_budget:
            break
        chunks.append(FILLER)

    hay = "".join(chunks)
    # Insert needle at character position ≈ depth * len
    pos = int(depth * max(len(hay) - 1, 0))
    # snap to sentence boundary-ish
    pos = hay.rfind(" ", 0, pos + 1)
    if pos < 0:
        pos = 0
    body = hay[:pos] + " " + NEEDLE + " " + hay[pos:]
    user = f"Context:\n{body}\n\nQuestion: {QUESTION}"
    messages = [{"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user + "\nAssistant:"

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        # Keep end (question) + needle region: middle truncate
        keep_head = target_tokens // 4
        keep_tail = target_tokens - keep_head
        ids = ids[:keep_head] + ids[-keep_tail:]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return text


def score_answer(text: str) -> dict:
    text_u = text.upper()
    hits = {k: (k.upper() in text_u or k in text) for k in NEEDLE_KEYS}
    # partial credit
    n = sum(1 for v in hits.values() if v)
    return {
        "hit_code": hits[NEEDLE_KEYS[0]],
        "hit_password": hits[NEEDLE_KEYS[1]],
        "hits": n,
        "success": n == len(NEEDLE_KEYS),
        "answer": text[:500],
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        print("Capping --ctx to 4096 (workstation policy).", flush=True)
        args.ctx = 4096

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"Model: {args.model}", flush=True)
    print(f"ctx={args.ctx} budget={args.budget} depths={depths}", flush=True)
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

    rows = []
    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        if am is not None:
            am = am.to(device)
        # Ensure needle survived truncation
        prompt_has_needle = all(k in prompt for k in NEEDLE_KEYS)
        print(
            f"\n=== depth={depth:.2f} actual_tokens={input_ids.shape[-1]} "
            f"needle_in_prompt={prompt_has_needle} ===",
            flush=True,
        )

        for method in methods:
            print(f"  {method}…", flush=True)
            try:
                past, logits = prefill_method(
                    model,
                    input_ids,
                    am,
                    method,
                    budget=args.budget,
                    window=args.window,
                    kernel=args.kernel,
                )
                tok_ids = greedy_generate(
                    model,
                    past,
                    logits,
                    args.max_new,
                    eos_id=tokenizer.eos_token_id,
                    # RoPE must continue from true prompt length after KV compress
                    next_position=int(input_ids.shape[-1]),
                )
                answer = tokenizer.decode(tok_ids, skip_special_tokens=True)
                sc = score_answer(answer)
                row = {
                    "model": args.model,
                    "method": method,
                    "ctx_target": args.ctx,
                    "ctx_actual": int(input_ids.shape[-1]),
                    "depth": depth,
                    "needle_in_prompt": prompt_has_needle,
                    "cache_tokens": cache_seq_len(past),
                    "cache_mb": round(cache_nbytes(past) / (1024**2), 2),
                    "budget": args.budget if method != "full" else -1,
                    **{k: sc[k] for k in ("hit_code", "hit_password", "hits", "success")},
                    "answer": sc["answer"].replace("\n", " "),
                }
                rows.append(row)
                print(
                    f"    success={row['success']} hits={row['hits']}/2 "
                    f"cache={row['cache_tokens']}  ans={row['answer'][:120]!r}",
                    flush=True,
                )
            except Exception as e:
                print(f"    ERROR {e}", flush=True)
                rows.append(
                    {
                        "model": args.model,
                        "method": method,
                        "depth": depth,
                        "ctx_target": args.ctx,
                        "error": str(e),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"needle_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "model": args.model,
        "ctx": args.ctx,
        "budget": args.budget,
        "depths": depths,
        "methods": methods,
        "needle": NEEDLE,
        "csv": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # summary
    print("\n=== summary (success rate) ===", flush=True)
    for method in methods:
        sub = [r for r in rows if r.get("method") == method and "success" in r]
        if not sub:
            continue
        rate = sum(1 for r in sub if r["success"]) / len(sub)
        print(f"  {method:12s}  {rate*100:.0f}%  ({sum(1 for r in sub if r['success'])}/{len(sub)})")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
