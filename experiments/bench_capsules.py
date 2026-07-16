"""
Fact-capsule vs seed_valley vs oracle — query-unknown streaming focus.

Arms @4k, multi-seed × depths:
  full
  oracle_r1          — H1′ upper bound (posthoc)
  stream_valley@B    — token/partial-valley stream (baseline)
  stream_capsules@B  — atomic capsule stream (new)

Primary question: does atomic packing help under query-unknown budgets
where seed_valley multi-seed fails at moderate B?

Usage:
  python experiments/bench_capsules.py
  python experiments/bench_capsules.py --seeds 0,1,2 --budgets 512,1024,1536
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
    find_all_critical_spans,
    span_recall,
)
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from capsules import prefill_streaming_capsules  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from scorer_valley import prefill_streaming_valley  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capsule vs valley streaming bench")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--budgets", default="512,1024,1536")
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--skip-full-oracle", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def decode(model, tokenizer, past, logits, seq_len, max_new):
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
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Capsule stream bench ===", flush=True)
    print(
        f"model={args.model} ctx={args.ctx} seeds={seeds} depths={depths} budgets={budgets}",
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

    for seed in seeds:
        for depth in depths:
            prompt = build_needle_prompt(tokenizer, args.ctx, depth, seed=seed)
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            seq_len = int(input_ids.shape[-1])
            critical = find_all_critical_spans(tokenizer, input_ids)
            print(
                f"\n--- seed={seed} depth={depth} seq={seq_len} n_crit={len(critical)} ---",
                flush=True,
            )

            if not args.skip_full_oracle:
                past, logits = prefill_chunked(model, input_ids, chunk_size=512)
                sc = decode(model, tokenizer, past, logits, seq_len, args.max_new)
                rows.append(
                    {
                        "arm": "full",
                        "seed": seed,
                        "depth": depth,
                        "budget": seq_len,
                        "success": sc["success"],
                        "hits": sc["hits"],
                        "cache_tokens": cache_seq_len(past),
                        "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                        "span_recall": 1.0,
                    }
                )
                print(f"  full: ok={sc['success']}", flush=True)
                del past

                past, logits = prefill_chunked(model, input_ids, chunk_size=512)
                keep = build_index_set(
                    seq_len,
                    sinks=8,
                    recent=args.window,
                    critical=critical,
                    mode="oracle_ctx",
                    span_context=args.R,
                )
                past = compress_keep_indices(past, keep)
                sc = decode(model, tokenizer, past, logits, seq_len, args.max_new)
                rows.append(
                    {
                        "arm": "oracle_r1",
                        "seed": seed,
                        "depth": depth,
                        "budget": len(keep),
                        "success": sc["success"],
                        "hits": sc["hits"],
                        "cache_tokens": cache_seq_len(past),
                        "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                        "span_recall": span_recall(keep, critical),
                    }
                )
                print(f"  oracle_r1: ok={sc['success']} n={len(keep)}", flush=True)
                del past
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for B in budgets:
                # valley stream
                try:
                    past, logits, st = prefill_streaming_valley(
                        model,
                        input_ids,
                        stream_budget=B,
                        final_budget=B,
                        chunk_size=512,
                        window_size=args.window,
                        sinks=8,
                        expand_radius=args.R,
                    )
                    sc = decode(model, tokenizer, past, logits, seq_len, args.max_new)
                    rows.append(
                        {
                            "arm": f"stream_valley@{B}",
                            "seed": seed,
                            "depth": depth,
                            "budget": B,
                            "success": sc["success"],
                            "hits": sc["hits"],
                            "cache_tokens": cache_seq_len(past),
                            "peak_cache": st.get("peak_cache_tokens", st.get("peak_cache")),
                            "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                            "answer": sc["answer"][:80].replace("\n", " "),
                        }
                    )
                    print(
                        f"  valley@{B}: ok={sc['success']} "
                        f"peak={st.get('peak_cache_tokens', st.get('peak_cache'))}",
                        flush=True,
                    )
                    del past
                except Exception as e:
                    rows.append(
                        {
                            "arm": f"stream_valley@{B}",
                            "seed": seed,
                            "depth": depth,
                            "budget": B,
                            "success": False,
                            "status": f"ERR:{type(e).__name__}",
                            "answer": str(e)[:100],
                        }
                    )
                    print(f"  valley@{B}: ERR {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # capsule stream
                try:
                    past, logits, st = prefill_streaming_capsules(
                        model,
                        input_ids,
                        stream_budget=B,
                        final_budget=B,
                        chunk_size=512,
                        window_size=args.window,
                        sinks=8,
                        expand_radius=args.R,
                    )
                    sc = decode(model, tokenizer, past, logits, seq_len, args.max_new)
                    rows.append(
                        {
                            "arm": f"stream_capsules@{B}",
                            "seed": seed,
                            "depth": depth,
                            "budget": B,
                            "success": sc["success"],
                            "hits": sc["hits"],
                            "cache_tokens": cache_seq_len(past),
                            "peak_cache": st.get("peak_cache"),
                            "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                            "n_caps_kept": st.get("last_pack", {}).get("n_kept"),
                            "n_caps_dropped": st.get("last_pack", {}).get("n_dropped"),
                            "answer": sc["answer"][:80].replace("\n", " "),
                        }
                    )
                    print(
                        f"  capsules@{B}: ok={sc['success']} peak={st.get('peak_cache')} "
                        f"caps={st.get('last_pack')}",
                        flush=True,
                    )
                    del past
                except Exception as e:
                    rows.append(
                        {
                            "arm": f"stream_capsules@{B}",
                            "seed": seed,
                            "depth": depth,
                            "budget": B,
                            "success": False,
                            "status": f"ERR:{type(e).__name__}",
                            "answer": str(e)[:100],
                        }
                    )
                    print(f"  capsules@{B}: ERR {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Aggregate
    def rate(arm: str) -> dict:
        sel = [r for r in rows if r.get("arm") == arm and r.get("status") is None]
        # also rows without status key
        sel = [r for r in rows if r.get("arm") == arm and not str(r.get("status", "")).startswith("ERR")]
        if not sel:
            return {"n": 0, "ok": 0, "rate": None}
        ok = sum(1 for r in sel if r.get("success"))
        return {"n": len(sel), "ok": ok, "rate": round(ok / len(sel), 4)}

    summary = {}
    for B in budgets:
        summary[f"stream_valley@{B}"] = rate(f"stream_valley@{B}")
        summary[f"stream_capsules@{B}"] = rate(f"stream_capsules@{B}")
    if not args.skip_full_oracle:
        summary["full"] = rate("full")
        summary["oracle_r1"] = rate("oracle_r1")

    # Compare capsule vs valley per budget
    wins = {}
    for B in budgets:
        v = summary[f"stream_valley@{B}"]["rate"]
        c = summary[f"stream_capsules@{B}"]["rate"]
        if v is None or c is None:
            wins[B] = "n/a"
        elif c > v:
            wins[B] = "capsules"
        elif v > c:
            wins[B] = "valley"
        else:
            wins[B] = "tie"

    print("\n=== Summary rates ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  winner_by_budget: {wins}", flush=True)

    # Verdict
    best_c = max(
        (summary[f"stream_capsules@{B}"]["rate"] or 0) for B in budgets
    )
    best_v = max((summary[f"stream_valley@{B}"]["rate"] or 0) for B in budgets)
    improved = any(
        (summary[f"stream_capsules@{B}"]["rate"] or 0)
        > (summary[f"stream_valley@{B}"]["rate"] or 0)
        for B in budgets
    )
    if improved and best_c >= 0.8:
        verdict = "CAPSULES_PROMISING"
    elif improved:
        verdict = "CAPSULES_EDGE"
    elif best_c >= best_v:
        verdict = "CAPSULES_TIE"
    else:
        verdict = "CAPSULES_NO_GAIN"

    print(f"\nVERDICT: {verdict}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"capsules_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "summary": summary,
        "winner_by_budget": wins,
        "model": args.model,
        "ctx": args.ctx,
        "seeds": seeds,
        "depths": depths,
        "budgets": budgets,
        "R": args.R,
        "csv": str(out),
        "idea": "atomic fact capsules under query-unknown stream",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
