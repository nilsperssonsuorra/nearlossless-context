"""
Cross-model transfer smoke: does H1 + streaming hold off the primary model?

Default transfer model: Qwen2.5-3B-Instruct (full attn, smaller).

Arms @4k mid (and optional 8k stream):
  full, oracle_r1, anti_oracle, stream@512, posthoc@176 (R=1)

Usage:
  python experiments/bench_transfer.py
  python experiments/bench_transfer.py --model meta-llama/Llama-3.2-3B-Instruct
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
from config import RESULTS_DIR, SNAPKV_WINDOW, TRANSFER_MODEL_ID  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from scorer_valley import compress_with_seed_valley, prefill_streaming_valley  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer smoke on second model")
    p.add_argument("--model", default=TRANSFER_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--also-8k", action="store_true", help="Also test stream@512 @8k mid")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new", type=int, default=48)
    return p.parse_args()


@torch.inference_mode()
def eval_arm(
    model,
    tokenizer,
    input_ids,
    arm: str,
    *,
    critical: list[int],
    window: int = SNAPKV_WINDOW,
    max_new: int = 48,
) -> dict:
    seq_len = int(input_ids.shape[-1])
    keep = None
    if arm == "full":
        past, logits = prefill_chunked(model, input_ids, chunk_size=512)
        keep = list(range(seq_len))
    elif arm == "oracle_r1":
        past, logits = prefill_chunked(model, input_ids, chunk_size=512)
        keep = build_index_set(
            seq_len,
            sinks=8,
            recent=window,
            critical=critical,
            mode="oracle_ctx",
            span_context=1,
        )
        past = compress_keep_indices(past, keep)
    elif arm == "anti_oracle":
        past, logits = prefill_chunked(model, input_ids, chunk_size=512)
        keep = build_index_set(
            seq_len,
            sinks=8,
            recent=window,
            critical=critical,
            mode="anti_oracle",
            span_context=1,
        )
        past = compress_keep_indices(past, keep)
    elif arm == "stream_512":
        past, logits, st = prefill_streaming_valley(
            model,
            input_ids,
            stream_budget=512,
            final_budget=512,
            chunk_size=512,
            window_size=window,
            sinks=8,
            expand_radius=1,
        )
    elif arm == "posthoc_176":
        past, logits = prefill_chunked(model, input_ids, chunk_size=512)
        past, keep = compress_with_seed_valley(
            model,
            input_ids,
            past,
            budget=176,
            window_size=window,
            sinks=8,
            expand_radius=1,
        )
    else:
        raise ValueError(arm)

    toks = greedy_generate(
        model,
        past,
        logits,
        max_new,
        eos_id=tokenizer.eos_token_id,
        next_position=seq_len,
    )
    sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
    return {
        "arm": arm,
        "success": sc["success"],
        "hits": sc["hits"],
        "span_recall": span_recall(keep, critical) if keep is not None else None,
        "cache_tokens": cache_seq_len(past),
        "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
        "answer": sc["answer"][:120].replace("\n", " "),
        "status": "ok",
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Transfer smoke ===", flush=True)
    print(f"Model={args.model} ctx={args.ctx} depth={args.depth}", flush=True)

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
    prompt = build_needle_prompt(tokenizer, args.ctx, args.depth)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    critical = find_all_critical_spans(tokenizer, input_ids)
    print(
        f"seq={input_ids.shape[-1]} n_crit={len(critical)}",
        flush=True,
    )

    arms = ["full", "oracle_r1", "anti_oracle", "stream_512", "posthoc_176"]
    for arm in arms:
        print(f"  {arm}…", flush=True)
        try:
            row = eval_arm(
                model,
                tokenizer,
                input_ids,
                arm,
                critical=critical,
                max_new=args.max_new,
            )
        except Exception as e:
            row = {
                "arm": arm,
                "success": False,
                "status": f"ERR:{type(e).__name__}",
                "answer": str(e)[:160],
            }
            print(f"    ERROR {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row.update(
            {
                "model": args.model,
                "ctx": args.ctx,
                "depth": args.depth,
            }
        )
        rows.append(row)
        if row.get("status") == "ok":
            print(
                f"    ok={row['success']} cache={row.get('cache_tokens')} "
                f"recall={row.get('span_recall')} ans={row.get('answer', '')[:50]!r}",
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.also_8k:
        print("\n=== 8k mid stream_512 ===", flush=True)
        p8 = build_needle_prompt(tokenizer, 8192, 0.5)
        ids8 = tokenizer(p8, return_tensors="pt")["input_ids"].to(device)
        crit8 = find_all_critical_spans(tokenizer, ids8)
        try:
            row = eval_arm(
                model,
                tokenizer,
                ids8,
                "stream_512",
                critical=crit8,
                max_new=args.max_new,
            )
            row.update({"model": args.model, "ctx": 8192, "depth": 0.5})
            rows.append(row)
            print(
                f"    ok={row['success']} cache={row.get('cache_tokens')} "
                f"ans={row.get('answer', '')[:50]!r}",
                flush=True,
            )
        except Exception as e:
            rows.append(
                {
                    "arm": "stream_512",
                    "ctx": 8192,
                    "success": False,
                    "status": f"ERR:{e}",
                    "model": args.model,
                }
            )
            print(f"    ERROR {e}", flush=True)

    def ok(arm: str, ctx: int | None = None) -> bool | None:
        for r in rows:
            if r.get("arm") != arm:
                continue
            if ctx is not None and r.get("ctx") != ctx:
                continue
            if r.get("status") != "ok":
                return False
            return bool(r.get("success"))
        return None

    full = ok("full", args.ctx)
    ora = ok("oracle_r1", args.ctx)
    anti = ok("anti_oracle", args.ctx)
    st = ok("stream_512", args.ctx)
    ph = ok("posthoc_176", args.ctx)

    if full and ora and anti is False and st and ph:
        verdict = "TRANSFER_OK"
        reasons = [
            "full+oracle+stream@512+posthoc@176 succeed; anti-oracle fails — H1+stream transfer"
        ]
    elif full and ora and anti is False and (st or ph):
        verdict = "TRANSFER_PARTIAL"
        reasons = ["H1 holds; some scorer arms fail — tune budget/R for this model"]
    elif full and (not ora or anti):
        verdict = "TRANSFER_H1_WEAK"
        reasons = ["suite or critical-span story weaker on this model"]
    elif not full:
        verdict = "TRANSFER_SUITE_FAIL"
        reasons = ["full mid-needle failed — chat template / scoring issue"]
    else:
        verdict = "TRANSFER_MIXED"
        reasons = ["see rows"]

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"transfer_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "model": args.model,
        "primary_for_comparison": "Qwen/Qwen3-4B-Instruct-2507",
        "csv": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
