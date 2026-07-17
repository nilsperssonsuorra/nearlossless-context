"""
Peak VRAM / latency / peak-cache table: full chunked vs sticky novelty stream@512.

Usage:
  python experiments/bench_systems_resources.py
  python experiments/bench_systems_resources.py --lengths 4096,16384,40960
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from novelty_detect import prefill_streaming_novelty_pin  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Systems resource table")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--lengths", default="4096,16384,40960")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def _vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**2), 1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Systems resource table ===", flush=True)
    print(f"lengths={lengths} budget={args.budget} depth={args.depth}", flush=True)

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

    for L in lengths:
        prompt = build_needle_prompt(tokenizer, L, args.depth, seed=args.seed)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])

        for arm in ("full", "novelty"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            if arm == "full":
                past, logits = prefill_chunked(model, input_ids, chunk_size=512)
                st = {"peak_cache": cache_seq_len(past)}
            else:
                past, logits, st = prefill_streaming_novelty_pin(
                    model,
                    tokenizer,
                    input_ids,
                    stream_budget=args.budget,
                    final_budget=args.budget,
                    chunk_size=512,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_s = time.perf_counter() - t0
            peak_vram = _vram_mb()
            peak_cache = st.get("peak_cache", cache_seq_len(past))
            final_cache = cache_seq_len(past)
            kv_mb = round(cache_nbytes(past) / (1024**2), 2)

            toks = greedy_generate(
                model,
                past,
                logits,
                args.max_new,
                eos_id=tokenizer.eos_token_id,
                next_position=seq_len,
            )
            sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))

            row = {
                "arm": arm if arm == "full" else f"novelty@{args.budget}",
                "L_target": L,
                "seq_len": seq_len,
                "success": sc["success"],
                "prefill_s": round(prefill_s, 3),
                "peak_cache": peak_cache,
                "final_cache": final_cache,
                "kv_mb": kv_mb,
                "peak_vram_mb": peak_vram,
            }
            rows.append(row)
            print(
                f"  L~{L} {row['arm']}: ok={sc['success']} "
                f"prefill={prefill_s:.2f}s peak_cache={peak_cache} "
                f"kv={kv_mb}MB vram={peak_vram}MB",
                flush=True,
            )
            del past
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"systems_resources_{ts}.json"
    out_csv = RESULTS_DIR / f"systems_resources_{ts}.csv"
    payload = {
        "ts": ts,
        "model": args.model,
        "budget": args.budget,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
