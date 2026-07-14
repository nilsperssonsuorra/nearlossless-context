"""
Measure L_ε: max context length with near-full-KV needle quality under fixed VRAM.

Uses **chunked prefill** so long prompts do not materialize one huge SDPA block
(workstation-safe vs single-shot 8k on WDDM).

Arms per length × depth:
  full           — full KV after chunked prefill (gold quality; may OOM)
  seed_valley@B  — chunked prefill → seed_valley compress to budget B
  snapkv@B       — SnapKV compress to B (uses its own full prefill path; may OOM)
  recent@B       — recent window control

Primary metric (ε=0 on needle):
  L_ε(M) = max L where M succeeds on all tested depths (or mid-depth if --mid-only)

Usage:
  python experiments/bench_l_epsilon.py
  python experiments/bench_l_epsilon.py --lengths 2048,4096,6144,8192 --allow-long
  python experiments/bench_l_epsilon.py --lengths 4096,6144 --budgets 176,256 --mid-only
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
    expand_critical,
    find_all_critical_spans,
    span_recall,
)
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate, prefill_method  # noqa: E402
from scorer_valley import compress_with_seed_valley  # noqa: E402
from snapkv import (  # noqa: E402
    cache_nbytes,
    cache_seq_len,
    compress_recent,
    prefill_chunked,
)
from utils import gpu_mem_mb, reset_peak_mem, timed_cuda, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L_ε ceiling: quality vs length under budget")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument(
        "--lengths",
        default="2048,4096,6144,8192",
        help="Context lengths to probe",
    )
    p.add_argument(
        "--allow-long",
        action="store_true",
        help="Allow L>4096 (needed for L_ε beyond workstation default)",
    )
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument(
        "--mid-only",
        action="store_true",
        help="Only mid-depth (0.5) — faster L_ε smoke",
    )
    p.add_argument(
        "--methods",
        default="full,seed_valley,snapkv,recent",
        help="Comma list: full, seed_valley, snapkv, recent",
    )
    p.add_argument(
        "--budgets",
        default="176,256",
        help="KV token budgets for compressed methods",
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
    window: int,
    chunk_size: int,
    sinks: int,
    R: int,
    max_new: int,
    critical: list[int],
    critical_r: list[int],
) -> dict:
    seq_len = int(input_ids.shape[-1])
    reset_peak_mem()
    keep: list[int] | None = None
    status = "ok"

    try:
        if method == "full":
            def _pf():
                return prefill_chunked(model, input_ids, chunk_size=chunk_size)

            (past, logits), prefill_s = timed_cuda(_pf)
            keep = list(range(seq_len))

        elif method == "seed_valley":
            def _pf():
                past0, logits0 = prefill_chunked(
                    model, input_ids, chunk_size=chunk_size
                )
                past0, keep0 = compress_with_seed_valley(
                    model,
                    input_ids,
                    past0,
                    budget=budget,
                    window_size=window,
                    sinks=sinks,
                    expand_radius=R,
                )
                return past0, logits0, keep0

            (past, logits, keep), prefill_s = timed_cuda(_pf)

        elif method == "snapkv":
            def _pf():
                return prefill_method(
                    model,
                    input_ids,
                    None,
                    "snapkv",
                    budget=budget,
                    window=window,
                    kernel=7,
                )

            (past, logits), prefill_s = timed_cuda(_pf)
            keep = None

        elif method == "recent":
            def _pf():
                past0, logits0 = prefill_chunked(
                    model, input_ids, chunk_size=chunk_size
                )
                past0 = compress_recent(past0, budget)
                return past0, logits0

            (past, logits), prefill_s = timed_cuda(_pf)
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
            "status": status,
            "success": sc["success"],
            "hits": sc["hits"],
            "prefill_s": round(prefill_s, 3),
            "cache_tokens": cache_seq_len(past),
            "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
            "vram_peak_mb": round(mem["max_allocated_mb"], 1),
            "span_recall_crit": span_recall(keep, critical) if keep is not None else None,
            "span_recall_crit_R": (
                span_recall(keep, critical_r) if keep is not None else None
            ),
            "keep_count": len(keep) if keep is not None else cache_seq_len(past),
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
            "span_recall_crit": None,
            "span_recall_crit_R": None,
            "keep_count": None,
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
            "span_recall_crit": None,
            "span_recall_crit_R": None,
            "keep_count": None,
            "answer": str(e)[:140],
        }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not args.allow_long:
        too = [L for L in lengths if L > 4096]
        if too:
            print(
                f"Dropping L>4096 {too} (pass --allow-long). "
                "Keeping ≤4096 only.",
                flush=True,
            )
            lengths = [L for L in lengths if L <= 4096]
    if args.mid_only:
        depths = [0.5]
    else:
        depths = [float(x) for x in args.depths.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== L_ε bench (chunked prefill + budgeted KV) ===", flush=True)
    print(f"Model={args.model}", flush=True)
    print(
        f"lengths={lengths} depths={depths} methods={methods} budgets={budgets}",
        flush=True,
    )
    print(f"chunk_size={args.chunk_size} window={args.window} R*={args.R}", flush=True)

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
        print(
            f"GPU after load: {torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )

    rows: list[dict] = []

    for L in lengths:
        for depth in depths:
            prompt = build_needle_prompt(tokenizer, L, depth)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            seq_len = int(input_ids.shape[-1])
            critical = find_all_critical_spans(tokenizer, input_ids)
            critical_r = expand_critical(critical, seq_len, args.R)
            print(
                f"\n=== L_target={L} actual={seq_len} depth={depth:.2f} "
                f"n_crit={len(critical)} ===",
                flush=True,
            )

            for method in methods:
                budget_list = [-1] if method == "full" else budgets
                for budget in budget_list:
                    if method != "full" and budget <= args.window:
                        print(f"  {method}@{budget} skip (≤window)", flush=True)
                        continue
                    tag = method if method == "full" else f"{method}@{budget}"
                    print(f"  {tag}…", flush=True)
                    row = run_arm(
                        model,
                        tokenizer,
                        input_ids,
                        method=method,
                        budget=max(budget, args.window + 16),
                        window=args.window,
                        chunk_size=args.chunk_size,
                        sinks=args.sink_size,
                        R=args.R,
                        max_new=args.max_new,
                        critical=critical,
                        critical_r=critical_r,
                    )
                    row.update(
                        {
                            "model": args.model,
                            "method": method,
                            "budget": budget if method != "full" else -1,
                            "ctx_target": L,
                            "ctx_actual": seq_len,
                            "depth": depth,
                            "R": args.R,
                            "chunk_size": args.chunk_size,
                        }
                    )
                    rows.append(row)
                    rc = row.get("span_recall_crit")
                    rc_s = f"{rc:.2f}" if isinstance(rc, float) else "n/a"
                    print(
                        f"    status={row['status']} success={row['success']} "
                        f"recall_c={rc_s} cache={row['cache_tokens']} "
                        f"kv_mb={row['kv_mb']} vram_peak={row['vram_peak_mb']} "
                        f"prefill={row['prefill_s']}s "
                        f"ans={row.get('answer', '')[:50]!r}",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # --- L_ε per method ---
    print("\n=== L_ε summary (ε=0 needle, all tested depths must pass) ===", flush=True)
    l_eps: dict[str, int | None] = {}
    method_keys: list[str] = []
    for method in methods:
        if method == "full":
            method_keys.append("full")
        else:
            for b in budgets:
                method_keys.append(f"{method}@{b}")

    for key in method_keys:
        if key == "full":
            method, budget = "full", -1
        else:
            method, b_s = key.rsplit("@", 1)
            budget = int(b_s)
        ok_lengths = []
        for L in lengths:
            sub = [
                r
                for r in rows
                if r.get("method") == method
                and r.get("budget") == budget
                and r.get("ctx_target") == L
                and r.get("status") == "ok"
            ]
            # require all depths present and success
            if not sub:
                continue
            depths_seen = {r["depth"] for r in sub}
            if set(depths) <= depths_seen and all(r.get("success") for r in sub):
                ok_lengths.append(L)
            else:
                # once failed at L, higher L may still work but L_ε is max contiguous from bottom
                pass
        # L_ε = max L with success; also require all smaller tested L succeed (monotone)
        l_star = None
        for L in lengths:
            if L in ok_lengths:
                l_star = L
            else:
                break
        l_eps[key] = l_star
        print(f"  {key:20s}  L_ε={l_star}  (ok_at={ok_lengths})", flush=True)

    full_l = l_eps.get("full")
    best_comp = None
    best_comp_l = -1
    for k, v in l_eps.items():
        if k == "full" or v is None:
            continue
        if v > best_comp_l:
            best_comp_l = v
            best_comp = k

    verdict = "INCONCLUSIVE"
    reasons = []
    if full_l is not None and best_comp is not None:
        if best_comp_l > full_l:
            verdict = "L_EPS_RAISED"
            reasons.append(
                f"{best_comp} reaches L_ε={best_comp_l} > full L_ε={full_l} "
                "(compression enables longer usable context under quality)"
            )
        elif best_comp_l == full_l:
            verdict = "L_EPS_MATCHED"
            reasons.append(
                f"best compressed {best_comp} matches full L_ε={full_l}; "
                "quality preserved with smaller decode KV "
                f"(check kv_mb columns)"
            )
        else:
            verdict = "L_EPS_BELOW_FULL"
            reasons.append(
                f"best compressed L_ε={best_comp_l} < full {full_l} — "
                "scorer fails at long L; improve detection or raise budget"
            )
    elif best_comp is not None and full_l is None:
        verdict = "L_EPS_RAISED"
        reasons.append(
            f"full OOM/fail; compressed {best_comp} still works to L_ε={best_comp_l}"
        )
    else:
        reasons.append("insufficient successful lengths")

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"l_epsilon_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "l_epsilon": l_eps,
        "lengths": lengths,
        "depths": depths,
        "methods": methods,
        "budgets": budgets,
        "chunk_size": args.chunk_size,
        "csv": str(out),
        "model": args.model,
        "hypothesis": "Budgeted KV (seed_valley) raises or matches L_ε vs full under 24GB",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
