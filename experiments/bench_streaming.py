"""
Streaming vs post-hoc compression: peak VRAM / peak cache tokens vs needle quality.

Arms:
  full_chunked       — chunked prefill, full KV (peak = L)
  posthoc_valley@B   — chunked full prefill → seed_valley at end (peak ≈ L during prefill)
  stream_valley@B    — online seed_valley after each chunk (peak ≈ B + chunk)
  stream_valley@B→F  — stream at B, final compress to F
  recent@B           — chunked prefill, keep last B only (control)

Goal: lower *peak* KV during prefill while matching post-hoc quality.

Usage:
  python experiments/bench_streaming.py
  python experiments/bench_streaming.py --lengths 4096,8192 --allow-long --mid-only
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

from bench_h1_oracle import find_all_critical_spans, span_recall  # noqa: E402
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from scorer_valley import (  # noqa: E402
    compress_with_seed_valley,
    prefill_streaming_valley,
)
from snapkv import (  # noqa: E402
    cache_nbytes,
    cache_seq_len,
    compress_recent,
    prefill_chunked,
)
from utils import gpu_mem_mb, reset_peak_mem, timed_cuda, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming valley peak-VRAM bench")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--lengths", default="4096,8192")
    p.add_argument("--allow-long", action="store_true")
    p.add_argument("--depths", default="0.5")
    p.add_argument("--mid-only", action="store_true")
    p.add_argument(
        "--methods",
        default="full_chunked,posthoc_valley,stream_valley,stream_hi,recent",
        help="full_chunked, posthoc_valley, stream_valley, stream_hi, recent",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=176,
        help="Final / stream budget for valley methods",
    )
    p.add_argument(
        "--stream-hi",
        type=int,
        default=512,
        help="stream_hi: intermediate stream budget before final@budget",
    )
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args()


@torch.inference_mode()
def run_arm(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    method: str,
    budget: int,
    stream_hi: int,
    window: int,
    chunk_size: int,
    sinks: int,
    R: int,
    max_new: int,
    critical: list[int],
) -> dict:
    seq_len = int(input_ids.shape[-1])
    reset_peak_mem()
    stats: dict = {}
    keep: list[int] | None = None

    try:
        if method == "full_chunked":
            def _pf():
                return prefill_chunked(model, input_ids, chunk_size=chunk_size)

            (past, logits), t = timed_cuda(_pf)
            stats = {
                "peak_cache_tokens": cache_seq_len(past),
                "n_compress": 0,
            }
            keep = list(range(seq_len))

        elif method == "posthoc_valley":
            def _pf():
                past0, logits0 = prefill_chunked(
                    model, input_ids, chunk_size=chunk_size
                )
                peak = cache_seq_len(past0)
                past0, keep0 = compress_with_seed_valley(
                    model,
                    input_ids,
                    past0,
                    budget=budget,
                    window_size=window,
                    sinks=sinks,
                    expand_radius=R,
                )
                return past0, logits0, keep0, peak

            (past, logits, keep, peak), t = timed_cuda(_pf)
            stats = {"peak_cache_tokens": peak, "n_compress": 1}

        elif method == "stream_valley":
            def _pf():
                return prefill_streaming_valley(
                    model,
                    input_ids,
                    stream_budget=budget,
                    final_budget=budget,
                    chunk_size=chunk_size,
                    window_size=window,
                    sinks=sinks,
                    expand_radius=R,
                )

            (past, logits, st), t = timed_cuda(_pf)
            stats = st
            keep = None

        elif method == "stream_hi":
            def _pf():
                return prefill_streaming_valley(
                    model,
                    input_ids,
                    stream_budget=stream_hi,
                    final_budget=budget,
                    chunk_size=chunk_size,
                    window_size=window,
                    sinks=sinks,
                    expand_radius=R,
                )

            (past, logits, st), t = timed_cuda(_pf)
            stats = st
            keep = None

        elif method == "recent":
            def _pf():
                past0, logits0 = prefill_chunked(
                    model, input_ids, chunk_size=chunk_size
                )
                peak = cache_seq_len(past0)
                past0 = compress_recent(past0, budget)
                return past0, logits0, peak

            (past, logits, peak), t = timed_cuda(_pf)
            stats = {"peak_cache_tokens": peak, "n_compress": 1}
            keep = list(range(max(0, seq_len - budget), seq_len))

        else:
            raise ValueError(method)

        toks = greedy_generate(
            model,
            past,
            logits,
            max_new,
            eos_id=tokenizer.eos_token_id,
            next_position=seq_len,
        )
        sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
        mem = gpu_mem_mb()
        row = {
            "status": "ok",
            "success": sc["success"],
            "hits": sc["hits"],
            "prefill_s": round(t, 3),
            "cache_tokens": cache_seq_len(past),
            "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
            "vram_peak_mb": round(mem["max_allocated_mb"], 1),
            "peak_cache_tokens": stats.get("peak_cache_tokens"),
            "n_compress": stats.get("n_compress"),
            "span_recall_crit": (
                span_recall(keep, critical) if keep is not None else None
            ),
            "answer": sc["answer"][:140].replace("\n", " "),
        }
        del past, logits, toks
        return row
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "success": False,
            "hits": 0,
            "prefill_s": None,
            "cache_tokens": None,
            "kv_mb": None,
            "vram_peak_mb": None,
            "peak_cache_tokens": None,
            "n_compress": None,
            "span_recall_crit": None,
            "answer": "",
        }
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": f"ERROR:{type(e).__name__}",
            "success": False,
            "hits": 0,
            "prefill_s": None,
            "cache_tokens": None,
            "kv_mb": None,
            "vram_peak_mb": None,
            "peak_cache_tokens": None,
            "n_compress": None,
            "span_recall_crit": None,
            "answer": str(e)[:160],
        }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not args.allow_long:
        too = [L for L in lengths if L > 4096]
        if too:
            print(f"Dropping L>4096 {too} (pass --allow-long).", flush=True)
            lengths = [L for L in lengths if L <= 4096]
    if args.mid_only or args.depths.strip() == "0.5":
        depths = [0.5]
    else:
        depths = [float(x) for x in args.depths.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Streaming valley: peak cache vs quality ===", flush=True)
    print(
        f"lengths={lengths} depths={depths} budget={args.budget} "
        f"stream_hi={args.stream_hi} chunk={args.chunk_size}",
        flush=True,
    )
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
        print(f"GPU after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    rows: list[dict] = []
    for L in lengths:
        for depth in depths:
            prompt = build_needle_prompt(tokenizer, L, depth)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            seq_len = int(input_ids.shape[-1])
            critical = find_all_critical_spans(tokenizer, input_ids)
            print(
                f"\n=== L={L} actual={seq_len} depth={depth:.2f} "
                f"n_crit={len(critical)} ===",
                flush=True,
            )
            for method in methods:
                print(f"  {method}…", flush=True)
                row = run_arm(
                    model,
                    tokenizer,
                    input_ids,
                    method=method,
                    budget=args.budget,
                    stream_hi=args.stream_hi,
                    window=args.window,
                    chunk_size=args.chunk_size,
                    sinks=args.sink_size,
                    R=args.R,
                    max_new=args.max_new,
                    critical=critical,
                )
                row.update(
                    {
                        "model": args.model,
                        "method": method,
                        "budget": args.budget,
                        "stream_hi": args.stream_hi,
                        "ctx_target": L,
                        "ctx_actual": seq_len,
                        "depth": depth,
                    }
                )
                rows.append(row)
                print(
                    f"    status={row['status']} success={row['success']} "
                    f"peak_cache={row['peak_cache_tokens']} "
                    f"final_cache={row['cache_tokens']} "
                    f"kv_mb={row['kv_mb']} vram_peak={row['vram_peak_mb']} "
                    f"prefill={row['prefill_s']}s "
                    f"ans={row.get('answer', '')[:48]!r}",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Summary
    print("\n=== Summary (success / peak_cache / vram) ===", flush=True)
    for method in methods:
        sub = [r for r in rows if r.get("method") == method and r.get("status") == "ok"]
        if not sub:
            print(f"  {method}: no ok rows", flush=True)
            continue
        ok = sum(1 for r in sub if r["success"])
        peaks = [r["peak_cache_tokens"] for r in sub if r.get("peak_cache_tokens")]
        vrams = [r["vram_peak_mb"] for r in sub if r.get("vram_peak_mb")]
        print(
            f"  {method:18s}  {ok}/{len(sub)} success  "
            f"peak_cache={peaks}  vram_peak={vrams}",
            flush=True,
        )

    # Verdict
    def all_ok(method: str) -> bool:
        sub = [r for r in rows if r.get("method") == method]
        return bool(sub) and all(
            r.get("status") == "ok" and r.get("success") for r in sub
        )

    post = [r for r in rows if r.get("method") == "posthoc_valley"]
    stream = [r for r in rows if r.get("method") == "stream_valley"]
    stream_hi = [r for r in rows if r.get("method") == "stream_hi"]
    full = [r for r in rows if r.get("method") == "full_chunked"]

    verdict = "INCONCLUSIVE"
    reasons = []
    if all_ok("stream_valley") and all_ok("posthoc_valley"):
        sp = max(r["peak_cache_tokens"] or 0 for r in stream)
        pp = max(r["peak_cache_tokens"] or 0 for r in post)
        if sp < pp * 0.5:
            verdict = "STREAM_PEAK_WIN"
            reasons.append(
                f"stream_valley matches posthoc quality with peak_cache "
                f"{sp} vs posthoc {pp} (~{pp/max(sp,1):.1f}× lower peak cache)"
            )
        else:
            verdict = "STREAM_QUALITY_OK"
            reasons.append("stream matches quality but peak cache not much lower")
    elif all_ok("stream_hi") and all_ok("posthoc_valley"):
        sp = max(r["peak_cache_tokens"] or 0 for r in stream_hi)
        pp = max(r["peak_cache_tokens"] or 0 for r in post)
        verdict = "STREAM_HI_WIN" if sp < pp * 0.6 else "STREAM_HI_OK"
        reasons.append(
            f"stream_hi@{args.stream_hi}→{args.budget}: quality ok; "
            f"peak_cache {sp} vs posthoc {pp}"
        )
        if not all_ok("stream_valley"):
            reasons.append(
                "tight stream_valley@budget failed quality — need higher stream budget"
            )
    elif all_ok("posthoc_valley") and not all_ok("stream_valley"):
        verdict = "STREAM_QUALITY_FAIL"
        reasons.append(
            "online compress drops critical spans before question arrives "
            "(expected risk for mid-needle)"
        )
    else:
        reasons.append("see per-row results")

    if full and all_ok("full_chunked"):
        fp = max(r["peak_cache_tokens"] or 0 for r in full)
        reasons.append(f"full peak_cache max={fp}")

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"streaming_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "budget": args.budget,
        "stream_hi": args.stream_hi,
        "lengths": lengths,
        "depths": depths,
        "csv": str(out),
        "model": args.model,
        "hypothesis": "Online seed_valley cuts peak KV vs post-hoc full→compress",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
