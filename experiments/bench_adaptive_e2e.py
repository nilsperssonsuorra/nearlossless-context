"""
End-to-end adaptive compress eval across task classes.

Tasks:
  single   — mid-depth single needle @4k
  multi3   — 3-needle recall_all @4k
  hop3     — 3-hop + distractors @4k

Arms per task:
  full
  posthoc_oracle_n   — adaptive with true n_entities
  posthoc_auto       — adaptive with peak-estimated n_entities
  stream_oracle_n    — stream adaptive with true n_entities
  stream_L_only      — stream adaptive n_entities=1 (L schedule only)

Usage:
  python experiments/bench_adaptive_e2e.py
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

from adaptive import policy_for  # noqa: E402
from bench_h3_hop3 import build_prompt as build_hop3  # noqa: E402
from bench_h3_hop3 import score_answer as score_hop3  # noqa: E402
from bench_h3_multi import NEEDLES, build_multi_prompt, score_multi  # noqa: E402
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from compress_adaptive import (  # noqa: E402
    prefill_posthoc_adaptive,
    prefill_stream_adaptive,
    score_prefix_from_past,
)
from adaptive import estimate_n_entities_from_scores  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new", type=int, default=96)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Adaptive E2E ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    needles = NEEDLES[:3]
    tasks = {
        "single": {
            "prompt": build_needle_prompt(tok, args.ctx, 0.5),
            "n_true": 1,
            "multi_hop": False,
            "score": lambda ans: score_answer(ans),
            "max_new": 48,
        },
        "multi3": {
            "prompt": build_multi_prompt(
                tok, args.ctx, needles, task="recall_all"
            )[0],
            "n_true": 3,
            "multi_hop": False,
            "score": lambda ans: score_multi(ans, needles, which="all"),
            "max_new": 96,
        },
        "hop3": {
            "prompt": build_hop3(tok, args.ctx),
            "n_true": 3,
            "multi_hop": True,
            "score": lambda ans: score_hop3(ans),
            "max_new": 48,
        },
    }

    rows = []
    for task_name, spec in tasks.items():
        ids = tok(spec["prompt"], return_tensors="pt")["input_ids"].to(device)
        T = int(ids.shape[-1])
        print(f"\n=== {task_name} seq={T} n_true={spec['n_true']} ===", flush=True)

        # peak estimate diagnostic
        past0, _ = prefill_chunked(model, ids, chunk_size=512)
        score, pref, _ = score_prefix_from_past(model, ids, past0)
        n_hat = estimate_n_entities_from_scores(score, prefix_len=pref)
        print(f"  peak n_entities_hat={n_hat}", flush=True)
        del past0
        torch.cuda.empty_cache()

        arms = [
            ("full", None, None),
            ("posthoc_oracle_n", spec["n_true"], False),
            ("posthoc_auto", None, False),
            ("stream_oracle_n", spec["n_true"], True),
            ("stream_L_only", 1, True),
        ]

        for arm, n_ent, is_stream in arms:
            print(f"  {arm}…", flush=True)
            try:
                if arm == "full":
                    past, logits = prefill_chunked(model, ids, chunk_size=512)
                    info = {"policy": None, "n_entities_hat": None}
                elif is_stream:
                    past, logits, info = prefill_stream_adaptive(
                        model,
                        ids,
                        multi_hop=spec["multi_hop"],
                        n_entities=n_ent,
                    )
                else:
                    past, logits, info = prefill_posthoc_adaptive(
                        model,
                        ids,
                        multi_hop=spec["multi_hop"],
                        n_entities=n_ent,
                    )

                toks = greedy_generate(
                    model,
                    past,
                    logits,
                    spec["max_new"],
                    eos_id=tok.eos_token_id,
                    next_position=T,
                )
                ans = tok.decode(toks, skip_special_tokens=True)
                sc = spec["score"](ans)
                ok = sc["success"]
                row = {
                    "task": task_name,
                    "arm": arm,
                    "success": ok,
                    "status": "ok",
                    "cache_tokens": cache_seq_len(past),
                    "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                    "n_true": spec["n_true"],
                    "n_hat": info.get("n_entities_hat"),
                    "policy": info.get("policy"),
                    "diag_n_hat": n_hat,
                    "answer": (sc.get("answer") or ans)[:160].replace("\n", " "),
                }
                pol = info.get("policy")
                pol_s = (
                    f"R={pol['R']} B={pol.get('budget')} S={pol.get('stream_budget')}"
                    if isinstance(pol, dict)
                    else "-"
                )
                print(
                    f"    ok={ok} cache={row['cache_tokens']} "
                    f"n_hat={row['n_hat']} {pol_s} "
                    f"ans={row['answer'][:40]!r}",
                    flush=True,
                )
                del past, logits, toks
            except Exception as e:
                row = {
                    "task": task_name,
                    "arm": arm,
                    "success": False,
                    "status": f"ERR:{e}",
                    "answer": str(e)[:120],
                }
                print(f"    ERROR {e}", flush=True)
            rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Verdict
    def ok(task: str, arm: str) -> bool:
        return any(
            r["task"] == task and r["arm"] == arm and r.get("success") for r in rows
        )

    print("\n=== Summary ===", flush=True)
    for t in tasks:
        for a in (
            "full",
            "posthoc_oracle_n",
            "posthoc_auto",
            "stream_oracle_n",
            "stream_L_only",
        ):
            print(f"  {t:8s} {a:18s}  {'PASS' if ok(t, a) else 'FAIL'}", flush=True)

    auto_ok = all(ok(t, "posthoc_auto") for t in tasks)
    oracle_n_ok = all(ok(t, "posthoc_oracle_n") for t in tasks)
    stream_n_ok = all(ok(t, "stream_oracle_n") for t in tasks)
    stream_l_ok = all(ok(t, "stream_L_only") for t in tasks)

    if auto_ok and stream_n_ok:
        verdict = "ADAPTIVE_E2E_OK"
        reasons = ["posthoc_auto + stream_oracle_n pass all tasks"]
    elif oracle_n_ok and not auto_ok:
        verdict = "ADAPTIVE_NEEDS_TRUE_N"
        reasons = ["oracle n_entities works; peak estimate mis-schedules"]
    elif auto_ok and not stream_l_ok:
        verdict = "ADAPTIVE_POSTHOC_OK_STREAM_NEEDS_N"
        reasons = [
            "posthoc auto ok; stream L-only fails some tasks (expected for multi/hop)"
        ]
    else:
        verdict = "ADAPTIVE_MIXED"
        reasons = ["see per-task summary"]

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"adaptive_e2e_{stamp}.csv"
    # flatten policy dict for csv
    flat = []
    for r in rows:
        rr = dict(r)
        pol = rr.pop("policy", None)
        if isinstance(pol, dict):
            rr["pol_R"] = pol.get("R")
            rr["pol_budget"] = pol.get("budget")
            rr["pol_stream"] = pol.get("stream_budget")
            rr["pol_note"] = pol.get("note")
        flat.append(rr)
    write_csv(out, flat)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "csv": str(out),
        "hypothesis": "Peak-estimated n_entities + lab schedule yields ε≈0 across tasks",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
