"""
H2: at fixed KV *bytes*, does critical±R* high-precision beat more tokens without priority?

Arms (per depth, L≈4k):
  full              — gold
  priority_bf16     — sinks ∪ crit±R* ∪ recent, bf16  (defines B0)
  volume_bf16       — same *token count* as priority, uniform/mid-heavy, no crit priority
  volume_int8       — ~2× tokens (int8 logical ≈ B0), uniform/mid-heavy, no crit priority
  volume_int8_avoid — max int8 tokens under B0 logical, **exclude** critical spans
  priority_int8     — same positions as priority_bf16 but int8 (precision ablation)

H2 support if:
  priority_bf16 ≈ full (ε=0) AND volume_int8_avoid fails AND
  (volume_bf16 or volume_int8) ≤ priority when they miss critical spans.

Usage:
  python experiments/bench_h2_bytes.py
  python experiments/bench_h2_bytes.py --depths 0.0,0.5,1.0 --R 1
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
    expand_critical,
    find_all_critical_spans,
    span_recall,
)
from bench_needle import build_needle_prompt, score_answer  # noqa: E402
from bytebudget import quant_dequant_int8  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, is_dynamic_cache  # noqa: E402
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2 equal-byte priority vs volume")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--R", type=int, default=1, help="Local context radius (H1′)")
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args()


def quantize_cache_int8(past) -> int:
    """In-place int8 fake-quant all layers. Returns logical nbytes."""
    logical = 0
    for layer in past.layers:
        if layer.keys is None:
            continue
        k, lk = quant_dequant_int8(layer.keys)
        v, lv = quant_dequant_int8(layer.values)
        layer.keys = k.contiguous()
        layer.values = v.contiguous()
        logical += lk + lv
    return logical


def uniform_keep(
    seq_len: int,
    n: int,
    *,
    sinks: int,
    recent: int,
    forbid: set[int] | None = None,
) -> list[int]:
    """sinks + recent + evenly spaced fillers; optional forbid (e.g. critical)."""
    forbid = forbid or set()
    n = min(n, seq_len)
    recent_idx = list(range(max(0, seq_len - recent), seq_len))
    sink_idx = list(range(min(sinks, seq_len)))
    base = [i for i in (set(sink_idx) | set(recent_idx)) if i not in forbid]
    if len(base) >= n:
        return sorted(base)[:n]
    need = n - len(base)
    # even spacing over full range, skip forbid and base
    chosen = set(base)
    # generate more candidates than needed
    if need > 0:
        step = max(seq_len / (need + 2), 1.0)
        cands = []
        x = step
        while x < seq_len and len(cands) < need * 4:
            i = int(x) % seq_len
            cands.append(i)
            x += step
        # also add middle-band dense fill
        mid = seq_len // 2
        for d in range(seq_len):
            cands.append((mid + d) % seq_len)
            cands.append((mid - d) % seq_len)
        for i in cands:
            if i in forbid or i in chosen:
                continue
            chosen.add(i)
            if len(chosen) >= n:
                break
    return sorted(chosen)[:n]


def bytes_per_token_bf16(past_full, seq_len: int) -> float:
    nb = cache_nbytes(past_full)
    return nb / max(seq_len, 1)


@torch.inference_mode()
def eval_keep(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    keep: list[int],
    *,
    use_int8: bool,
    max_new: int,
    critical: list[int],
    arm: str,
) -> dict:
    seq_len = int(input_ids.shape[-1])
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].clone()
    del out
    past = compress_keep_indices(past, keep)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logical = cache_nbytes(past)
    if use_int8:
        logical = quantize_cache_int8(past)
    runtime = cache_nbytes(past)
    cache_toks = cache_seq_len(past)
    toks = greedy_generate(
        model,
        past,
        logits,
        max_new,
        eos_id=tokenizer.eos_token_id,
        next_position=seq_len,
    )
    ans = tokenizer.decode(toks, skip_special_tokens=True)
    sc = score_answer(ans)
    del past, logits, toks
    return {
        "arm": arm,
        "success": sc["success"],
        "hits": sc["hits"],
        "span_recall": span_recall(keep, critical),
        "keep_count": len(keep),
        "cache_tokens": cache_toks,
        "runtime_kv_mb": round(runtime / (1024**2), 3),
        "logical_kv_mb": round(logical / (1024**2), 3),
        "use_int8": use_int8,
        "answer": sc["answer"][:160].replace("\n", " "),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== H2 equal-byte: priority vs volume ===", flush=True)
    print(f"Model={args.model} ctx={args.ctx} R*={args.R}", flush=True)
    print(
        "priority = sinks ∪ crit±R ∪ recent (bf16); "
        "volume = more/equal tokens without crit priority",
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU after load: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
            flush=True,
        )

    rows: list[dict] = []

    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])
        critical = find_all_critical_spans(tokenizer, input_ids)
        crit_set = set(critical)

        print(
            f"\n=== depth={depth:.2f} seq={seq_len} n_crit={len(critical)} ===",
            flush=True,
        )

        # full (then free KV so arms can prefill without holding 2× cache)
        print("  full…", flush=True)
        out = model(input_ids=input_ids, use_cache=True)
        past_f, logits_f = out.past_key_values, out.logits[:, -1, :]
        del out
        bpt = bytes_per_token_bf16(past_f, seq_len)
        kv_mb_full = round(cache_nbytes(past_f) / (1024**2), 3)
        cache_toks_full = cache_seq_len(past_f)
        toks = greedy_generate(
            model,
            past_f,
            logits_f,
            args.max_new,
            eos_id=tokenizer.eos_token_id,
            next_position=seq_len,
        )
        sc = score_answer(tokenizer.decode(toks, skip_special_tokens=True))
        del past_f, logits_f, toks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row_full = {
            "model": args.model,
            "depth": depth,
            "arm": "full",
            "success": sc["success"],
            "hits": sc["hits"],
            "span_recall": 1.0,
            "keep_count": seq_len,
            "cache_tokens": cache_toks_full,
            "runtime_kv_mb": kv_mb_full,
            "logical_kv_mb": kv_mb_full,
            "use_int8": False,
            "B0_mb": None,
            "bytes_per_tok_bf16": round(bpt, 1),
            "answer": sc["answer"][:160].replace("\n", " "),
            "ctx_actual": seq_len,
        }
        rows.append(row_full)
        print(
            f"    success={sc['success']} kv_mb={row_full['runtime_kv_mb']} "
            f"B/token≈{bpt:.0f}",
            flush=True,
        )
        if not sc["success"]:
            print("  full failed — skip depth", flush=True)
            continue

        # priority set
        keep_pri = build_index_set(
            seq_len,
            sinks=args.sink_size,
            recent=args.recent_window,
            critical=critical,
            mode="oracle_ctx",
            span_context=args.R,
        )
        n_pri = len(keep_pri)

        arms_spec = []

        # 1) priority bf16
        arms_spec.append(("priority_bf16", keep_pri, False))

        # 2) volume bf16 same token count, no priority (may hit crit by chance)
        keep_vol_bf = uniform_keep(
            seq_len,
            n_pri,
            sinks=args.sink_size,
            recent=args.recent_window,
            forbid=None,
        )
        arms_spec.append(("volume_bf16_same_n", keep_vol_bf, False))

        # 3) volume bf16 same n, avoid critical
        keep_vol_bf_av = uniform_keep(
            seq_len,
            n_pri,
            sinks=args.sink_size,
            recent=max(32, args.recent_window // 2),
            forbid=crit_set,
        )
        # if recent overlapped crit at depth=1, we already forbade — shrink recent
        arms_spec.append(("volume_bf16_avoid_crit", keep_vol_bf_av, False))

        # 4) volume int8: ~2x tokens (logical int8 ~ half bf16 per token)
        # int8 logical ≈ 1 byte * elems + 2 * scales; roughly ~0.5x bf16 for large D
        n_int8 = min(seq_len, int(n_pri * 2.0))
        keep_vol_i8 = uniform_keep(
            seq_len,
            n_int8,
            sinks=args.sink_size,
            recent=args.recent_window,
            forbid=None,
        )
        arms_spec.append(("volume_int8_2x_n", keep_vol_i8, True))

        # 5) volume int8 avoid crit, 2x n
        keep_vol_i8_av = uniform_keep(
            seq_len,
            n_int8,
            sinks=args.sink_size,
            recent=max(32, args.recent_window // 2),
            forbid=crit_set,
        )
        arms_spec.append(("volume_int8_2x_avoid", keep_vol_i8_av, True))

        # 6) priority positions but int8
        arms_spec.append(("priority_int8", keep_pri, True))

        B0 = None
        for arm_name, keep, use_i8 in arms_spec:
            print(f"  {arm_name} (n={len(keep)}, int8={use_i8})…", flush=True)
            try:
                row = eval_keep(
                    model,
                    tokenizer,
                    input_ids,
                    keep,
                    use_int8=use_i8,
                    max_new=args.max_new,
                    critical=critical,
                    arm=arm_name,
                )
                if arm_name == "priority_bf16":
                    B0 = row["logical_kv_mb"]
                row["model"] = args.model
                row["depth"] = depth
                row["ctx_actual"] = seq_len
                row["B0_mb"] = B0
                row["R"] = args.R
                row["bytes_per_tok_bf16"] = round(bpt, 1)
                rows.append(row)
                print(
                    f"    success={row['success']} recall={row['span_recall']:.2f} "
                    f"log_mb={row['logical_kv_mb']} run_mb={row['runtime_kv_mb']} "
                    f"ans={row['answer'][:60]!r}",
                    flush=True,
                )
            except Exception as e:
                print(f"    ERROR {e}", flush=True)
                rows.append(
                    {
                        "arm": arm_name,
                        "depth": depth,
                        "error": str(e),
                        "model": args.model,
                    }
                )
            torch.cuda.empty_cache()

    # Verdict
    print("\n=== H2 summary (success rate by arm) ===", flush=True)
    arms = sorted({r["arm"] for r in rows if "arm" in r and "success" in r})
    summary = {}
    for arm in arms:
        sub = [r for r in rows if r.get("arm") == arm and "success" in r]
        ok = sum(1 for r in sub if r["success"])
        summary[arm] = {"ok": ok, "n": len(sub), "rate": ok / len(sub) if sub else None}
        print(f"  {arm:28s}  {100*ok/len(sub):.0f}% ({ok}/{len(sub)})", flush=True)

    pri = summary.get("priority_bf16", {}).get("rate")
    full = summary.get("full", {}).get("rate")
    avoid_bf = summary.get("volume_bf16_avoid_crit", {}).get("rate")
    avoid_i8 = summary.get("volume_int8_2x_avoid", {}).get("rate")
    vol_i8 = summary.get("volume_int8_2x_n", {}).get("rate")
    pri_i8 = summary.get("priority_int8", {}).get("rate")

    verdict = "INCONCLUSIVE"
    reasons = []
    if full == 1.0 and pri == 1.0:
        if (avoid_bf == 0.0 or avoid_bf is None) and (avoid_i8 == 0.0 or avoid_i8 is None):
            if pri_i8 == 1.0:
                verdict = "H2_SUPPORTED"
                reasons.append(
                    "priority±R* at bf16 (and int8) matches full; "
                    "equal/double token volume avoiding critical fails — "
                    "bytes spent on critical local context beat more irrelevant tokens"
                )
            elif pri_i8 is not None and pri_i8 < 1.0:
                verdict = "H2_PRECISION_MATTERS"
                reasons.append(
                    "priority positions work in bf16 but int8 hurts — "
                    "need high precision on critical spans, not only positions"
                )
            else:
                verdict = "H2_SUPPORTED_WEAK"
                reasons.append("priority works; avoid-crit volume fails")
        elif (vol_i8 or 0) >= (pri or 0) and (avoid_i8 or 0) > 0:
            verdict = "H2_WEAK_OR_FALSE"
            reasons.append("volume without priority still often works — H2 weakened")
        else:
            verdict = "H2_MIXED"
            reasons.append("see per-depth rows")
    elif pri is not None and pri < 1.0:
        verdict = "PRIORITY_BASELINE_FAIL"
        reasons.append("priority_bf16 not at full quality — fix R or suite first")
    else:
        reasons.append("missing rates")

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h2_bytes_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "hypothesis": "H2 equal-byte critical±R* high-precision beats volume without priority",
        "R": args.R,
        "csv": str(out),
        "model": args.model,
        "depths": depths,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
