"""
Long-L multi-seed: novelty discovery vs attention valley streaming.

Question: does surface novelty keep ε≈0 at stream@512 when L grows to
8k–16k (where mid-depth valley often needed 1536 under fixed-filler
sweeps, and multi-seed@4k valley@512 only hit 33%)?

Usage:
  python experiments/bench_novelty_longL.py
  python experiments/bench_novelty_longL.py --ctx 8192,16384 --seeds 0,1,2 --budgets 512,1536
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

from bench_h1_oracle import find_all_critical_spans  # noqa: E402
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from capsules import prefill_streaming_oracle_pin  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from novelty_detect import prefill_streaming_novelty_pin  # noqa: E402
from scorer_valley import prefill_streaming_valley  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-L multi-seed novelty vs valley")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", default="8192,12288,16384", help="Comma list of target L")
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budgets", default="512,1536")
    p.add_argument(
        "--arms",
        default="valley,novelty",
        help="Comma list: valley,novelty,oracle_pin",
    )
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--chunk", type=int, default=512)
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
    ctxs = [int(x) for x in args.ctx.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Long-L novelty multi-seed ===", flush=True)
    print(
        f"model={args.model} ctx={ctxs} seeds={seeds} depths={depths} "
        f"budgets={budgets} arms={arms}",
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

    for ctx in ctxs:
        for seed in seeds:
            for depth in depths:
                prompt = build_needle_prompt(tokenizer, ctx, depth, seed=seed)
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    device
                )
                seq_len = int(input_ids.shape[-1])
                critical = find_all_critical_spans(tokenizer, input_ids)
                print(
                    f"\n--- L~{ctx} seed={seed} depth={depth} seq={seq_len} "
                    f"n_crit={len(critical)} ---",
                    flush=True,
                )

                for B in budgets:
                    for arm in arms:
                        try:
                            if arm == "valley":
                                past, logits, st = prefill_streaming_valley(
                                    model,
                                    input_ids,
                                    stream_budget=B,
                                    final_budget=B,
                                    chunk_size=args.chunk,
                                    window_size=args.window,
                                    sinks=8,
                                    expand_radius=args.R,
                                )
                                extra = {}
                            elif arm == "novelty":
                                past, logits, st = prefill_streaming_novelty_pin(
                                    model,
                                    tokenizer,
                                    input_ids,
                                    stream_budget=B,
                                    final_budget=B,
                                    chunk_size=args.chunk,
                                    window_size=args.window,
                                    sinks=8,
                                    expand_radius=args.R,
                                )
                                extra = {
                                    "novelty_ret_proxy": st.get(
                                        "last_novelty_retention_proxy"
                                    ),
                                    "n_novelty_kept": st.get("last_n_novelty_kept"),
                                }
                            elif arm == "oracle_pin":
                                past, logits, st = prefill_streaming_oracle_pin(
                                    model,
                                    input_ids,
                                    critical=critical,
                                    stream_budget=B,
                                    final_budget=B,
                                    chunk_size=args.chunk,
                                    window_size=args.window,
                                    sinks=8,
                                    expand_radius=args.R,
                                )
                                extra = {
                                    "oracle_retention": st.get("oracle_retention"),
                                }
                            else:
                                raise ValueError(f"unknown arm {arm}")

                            sc = decode(
                                model, tokenizer, past, logits, seq_len, args.max_new
                            )
                            peak = st.get("peak_cache_tokens", st.get("peak_cache"))
                            row = {
                                "arm": f"stream_{arm}@{B}",
                                "arm_base": arm,
                                "ctx_target": ctx,
                                "seq_len": seq_len,
                                "seed": seed,
                                "depth": depth,
                                "budget": B,
                                "success": sc["success"],
                                "hits": sc["hits"],
                                "cache_tokens": cache_seq_len(past),
                                "peak_cache": peak,
                                "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                                "answer": sc["answer"][:100].replace("\n", " "),
                                **extra,
                            }
                            rows.append(row)
                            print(
                                f"  {arm}@{B}: ok={sc['success']} peak={peak} "
                                f"hits={sc['hits']}",
                                flush=True,
                            )
                            del past
                        except Exception as e:
                            rows.append(
                                {
                                    "arm": f"stream_{arm}@{B}",
                                    "arm_base": arm,
                                    "ctx_target": ctx,
                                    "seq_len": seq_len,
                                    "seed": seed,
                                    "depth": depth,
                                    "budget": B,
                                    "success": False,
                                    "status": f"ERR:{type(e).__name__}",
                                    "answer": str(e)[:120],
                                }
                            )
                            print(f"  {arm}@{B}: ERR {e}", flush=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # Aggregate
    print("\n=== SUMMARY (success rate) ===", flush=True)
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for r in rows:
        key = (r.get("ctx_target"), r.get("arm"), r.get("budget"))
        groups[key].append(bool(r.get("success")))

    summary_rows = []
    for (ctx, arm, B), oks in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        n = len(oks)
        s = sum(oks)
        rate = s / n if n else 0.0
        # depth breakdown
        depth_ok: dict[float, list[bool]] = defaultdict(list)
        for r in rows:
            if (
                r.get("ctx_target") == ctx
                and r.get("arm") == arm
                and r.get("budget") == B
            ):
                depth_ok[float(r["depth"])].append(bool(r.get("success")))
        by_d = {
            d: f"{sum(v)}/{len(v)}" for d, v in sorted(depth_ok.items())
        }
        print(
            f"  L={ctx} {arm}: {s}/{n} ({100*rate:.0f}%) depths={by_d}",
            flush=True,
        )
        summary_rows.append(
            {
                "ctx": ctx,
                "arm": arm,
                "budget": B,
                "success": s,
                "n": n,
                "rate": round(rate, 4),
                "depths": by_d,
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"novelty_longL_{ts}.json"
    out_csv = RESULTS_DIR / f"novelty_longL_{ts}.csv"
    payload = {
        "ts": ts,
        "model": args.model,
        "ctxs": ctxs,
        "seeds": seeds,
        "depths": depths,
        "budgets": budgets,
        "arms": arms,
        "summary": summary_rows,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
