"""
H1 kill experiment: critical-span retention necessity + sufficiency.

Arms:
  full         — full KV (gold)
  oracle       — sinks + critical spans + recent window only
  anti_oracle  — same token budget as oracle, but spans excluded
  recent       — recent window only (control)
  snapkv       — production-ish baseline (optional correlation)

See RESEARCH_BRIEF.md §3–4.

Usage:
  python experiments/bench_h1_oracle.py
  python experiments/bench_h1_oracle.py --ctx 4096 --depths 0.0,0.5,1.0
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

from bench_needle import (  # noqa: E402
    NEEDLE,
    NEEDLE_KEYS,
    QUESTION,
    build_needle_prompt,
    score_answer,
)
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate, prefill_method  # noqa: E402
from snapkv import (  # noqa: E402
    cache_nbytes,
    cache_seq_len,
    is_dynamic_cache,
)
from utils import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1 oracle / anti-oracle kill experiment")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument(
        "--arms",
        default="full,oracle,oracle_ctx,anti_oracle,recent,snapkv",
        help="Comma list of arms to run",
    )
    p.add_argument(
        "--span-context",
        type=int,
        default=16,
        help="For oracle_ctx: keep ±R tokens around each critical index",
    )
    return p.parse_args()


def find_minimal_span(tokenizer, ids: list[int], needle: str) -> list[int]:
    """Smallest [i, i+L) whose decode contains `needle`."""
    n = len(ids)
    best: list[int] | None = None
    for i in range(n):
        for L in range(1, min(48, n - i + 1)):
            piece = tokenizer.decode(ids[i : i + L], skip_special_tokens=False)
            if needle in piece:
                span = list(range(i, i + L))
                if best is None or len(span) < len(best):
                    best = span
                break
    return best or []


def find_all_critical_spans(tokenizer, input_ids: torch.Tensor) -> list[int]:
    ids = input_ids[0].tolist()
    spans: set[int] = set()
    for key in NEEDLE_KEYS:
        for t in find_minimal_span(tokenizer, ids, key):
            spans.add(t)
    # also whole NEEDLE sentence if present
    for t in find_minimal_span(tokenizer, ids, "BLUE-ORBIT"):
        spans.add(t)
    for t in find_minimal_span(tokenizer, ids, "maple-quartz"):
        spans.add(t)
    return sorted(spans)


def expand_critical(critical: list[int], seq_len: int, radius: int) -> list[int]:
    out: set[int] = set()
    for i in critical:
        for j in range(i - radius, i + radius + 1):
            if 0 <= j < seq_len:
                out.add(j)
    return sorted(out)


def build_index_set(
    seq_len: int,
    *,
    sinks: int,
    recent: int,
    critical: list[int],
    mode: str,
    span_context: int = 16,
) -> list[int]:
    """
    mode:
      oracle — sinks ∪ critical ∪ recent
      oracle_ctx — sinks ∪ critical±R ∪ recent
      anti_oracle — same count as oracle, but no critical; fill from mid non-critical
      recent — last `recent` only
    """
    recent_idx = list(range(max(0, seq_len - recent), seq_len))
    sink_idx = list(range(min(sinks, seq_len)))
    crit = [i for i in critical if 0 <= i < seq_len]

    if mode == "recent":
        return sorted(set(recent_idx))

    if mode == "oracle":
        return sorted(set(sink_idx) | set(crit) | set(recent_idx))

    if mode == "oracle_ctx":
        crit_x = expand_critical(crit, seq_len, span_context)
        return sorted(set(sink_idx) | set(crit_x) | set(recent_idx))

    if mode == "anti_oracle":
        crit_set = set(crit)
        oracle_set = set(sink_idx) | crit_set | set(recent_idx)
        budget = len(oracle_set)
        # Exclude critical tokens even if they fall inside the recent window
        # (depth≈1.0 places the needle at the end).
        base = (set(sink_idx) | set(recent_idx)) - crit_set
        forbidden = crit_set
        cands = [i for i in range(seq_len) if i not in base and i not in forbidden]
        cands.sort(key=lambda i: abs(i - seq_len // 2))
        chosen = set(base)
        for i in cands:
            if len(chosen) >= budget:
                break
            chosen.add(i)
        if len(chosen) < budget:
            for i in range(seq_len):
                if i not in forbidden and i not in chosen:
                    chosen.add(i)
                if len(chosen) >= budget:
                    break
        chosen -= crit_set
        return sorted(chosen)

    raise ValueError(mode)


def compress_keep_indices(past, keep: list[int]):
    """
    Keep only listed positions (sorted) in full-attention layers. Mutates cache.

    Hybrid models (e.g. Gemma-4): sliding-window layers already hold a local
    window and their attention masks depend on cumulative_length — do not
    index_select them. Long-range fact retention lives in full layers.
    """
    from snapkv import is_sliding_layer  # local import avoids cycles

    if not is_dynamic_cache(past):
        raise TypeError(type(past))
    if not keep:
        raise ValueError("empty keep set")
    device = None
    for layer in past.layers:
        if layer.keys is not None:
            device = layer.keys.device
            break
    if device is None:
        raise ValueError("empty cache")
    keep_t = torch.tensor(sorted(set(int(i) for i in keep)), device=device, dtype=torch.long)
    for layer in past.layers:
        if layer.keys is None:
            continue
        if is_sliding_layer(layer):
            continue
        s = int(layer.keys.shape[-2])
        # Guard: keep indices must be in [0, S)
        if int(keep_t.max()) >= s or int(keep_t.min()) < 0:
            valid = keep_t[(keep_t >= 0) & (keep_t < s)]
            if valid.numel() == 0:
                # Fall back to recent tail
                n = min(len(keep), s)
                valid = torch.arange(s - n, s, device=device, dtype=torch.long)
            idx = valid
        else:
            idx = keep_t
        layer.keys = layer.keys.index_select(2, idx).contiguous()
        layer.values = layer.values.index_select(2, idx).contiguous()
    return past


def span_recall(keep: list[int], critical: list[int]) -> float:
    if not critical:
        return 1.0
    ks = set(keep)
    return sum(1 for t in critical if t in ks) / len(critical)


@torch.inference_mode()
def run_arm(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    arm: str,
    *,
    critical: list[int],
    sink_size: int,
    recent_window: int,
    max_new: int,
    span_context: int = 16,
) -> dict:
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device

    if arm == "full":
        out = model(input_ids=input_ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        keep = list(range(seq_len))
    elif arm == "snapkv":
        past, logits = prefill_method(
            model,
            input_ids,
            None,
            "snapkv",
            budget=max(512, sink_size + recent_window + len(critical) + 64),
            window=recent_window,
            kernel=7,
        )
        # cannot easily know keep set; mark recall unknown
        keep = []
        recall = None
    elif arm in ("oracle", "oracle_ctx", "anti_oracle", "recent"):
        out = model(input_ids=input_ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        keep = build_index_set(
            seq_len,
            sinks=sink_size,
            recent=recent_window,
            critical=critical,
            mode=arm if arm != "oracle_ctx" else "oracle_ctx",
            span_context=span_context,
        )
        past = compress_keep_indices(past, keep)
    else:
        raise ValueError(arm)

    if arm != "snapkv":
        recall = span_recall(keep, critical)
    else:
        recall = None

    toks = greedy_generate(
        model,
        past,
        logits,
        max_new,
        eos_id=tokenizer.eos_token_id,
        next_position=seq_len,
    )
    answer = tokenizer.decode(toks, skip_special_tokens=True)
    sc = score_answer(answer)

    return {
        "arm": arm,
        "success": sc["success"],
        "hits": sc["hits"],
        "span_recall": recall,
        "cache_tokens": cache_seq_len(past),
        "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
        "keep_count": len(keep) if keep else None,
        "n_critical": len(critical),
        "answer": sc["answer"][:200].replace("\n", " "),
    }


def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        print("Capping ctx to 4096 for interactive safety.", flush=True)
        args.ctx = 4096

    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== H1 kill experiment ===", flush=True)
    print(f"Model={args.model} ctx={args.ctx} depths={depths}", flush=True)
    print(f"Arms={arms}", flush=True)
    print(
        "H1 support: oracle≈full success AND anti_oracle≈fail AND recall correlates.",
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

    rows = []
    for depth in depths:
        prompt = build_needle_prompt(tokenizer, args.ctx, depth)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        critical = find_all_critical_spans(tokenizer, input_ids)
        print(
            f"\n=== depth={depth:.2f} seq={input_ids.shape[-1]} "
            f"n_critical_tokens={len(critical)} ===",
            flush=True,
        )
        if not critical:
            print("  WARNING: no critical spans found — suite broken", flush=True)

        for arm in arms:
            print(f"  {arm}…", flush=True)
            try:
                row = run_arm(
                    model,
                    tokenizer,
                    input_ids,
                    arm,
                    critical=critical,
                    sink_size=args.sink_size,
                    recent_window=args.recent_window,
                    max_new=args.max_new,
                    span_context=args.span_context,
                )
                row.update(
                    {
                        "model": args.model,
                        "depth": depth,
                        "ctx_target": args.ctx,
                        "ctx_actual": int(input_ids.shape[-1]),
                    }
                )
                rows.append(row)
                print(
                    f"    success={row['success']} hits={row['hits']} "
                    f"recall={row['span_recall']} cache={row['cache_tokens']} "
                    f"kv_mb={row['kv_mb']} ans={row['answer'][:80]!r}",
                    flush=True,
                )
            except Exception as e:
                print(f"    ERROR {e}", flush=True)
                rows.append(
                    {
                        "arm": arm,
                        "depth": depth,
                        "error": str(e),
                        "model": args.model,
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- verdict ---
    def rate(arm: str) -> tuple[float | None, int, int]:
        sub = [r for r in rows if r.get("arm") == arm and "success" in r]
        if not sub:
            return None, 0, 0
        ok = sum(1 for r in sub if r["success"])
        return ok / len(sub), ok, len(sub)

    print("\n=== H1 verdict (smoke) ===", flush=True)
    summary = {}
    for arm in arms:
        r, ok, n = rate(arm)
        summary[arm] = {"rate": r, "ok": ok, "n": n}
        if r is not None:
            print(f"  {arm:12s}  {100*r:.0f}%  ({ok}/{n})", flush=True)

    full_r = summary.get("full", {}).get("rate")
    ora_r = summary.get("oracle", {}).get("rate")
    ora_ctx_r = summary.get("oracle_ctx", {}).get("rate")
    anti_r = summary.get("anti_oracle", {}).get("rate")

    verdict = "INCONCLUSIVE"
    reasons = []
    if full_r is not None and full_r < 1.0:
        verdict = "SUITE_INVALID"
        reasons.append("full KV not 100% — fix prompt/suite before testing H1")
    elif ora_r is not None and anti_r is not None and full_r == 1.0:
        if ora_r < 1.0:
            if ora_ctx_r is not None and ora_ctx_r >= 1.0:
                verdict = "H1_NEEDS_LOCAL_CONTEXT"
                reasons.append(
                    "bare critical spans insufficient; spans±context restores full — "
                    "revise H1 to include local context window around facts"
                )
            else:
                verdict = "H1_SUFFICIENCY_FAIL"
                reasons.append("oracle-keep < full — retaining spans not enough")
        elif anti_r > 0.0:
            if anti_r >= 0.5:
                verdict = "H1_NECESSITY_FAIL"
                reasons.append("anti-oracle often succeeds — fact recoverable without spans")
            else:
                verdict = "H1_SUPPORTED_WEAK"
                reasons.append("oracle ok; anti-oracle rare success — check leaks")
        else:
            verdict = "H1_SUPPORTED"
            reasons.append("oracle≈full and anti-oracle fails — H1 supported on this smoke")
    else:
        reasons.append("missing arms or rates")

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h1_oracle_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "hypothesis": "H1 critical-span necessity+sufficiency for single-fact retrieval",
        "csv": str(out),
        "depths": depths,
        "arms": arms,
        "ctx": args.ctx,
        "model": args.model,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print(f"Wrote {out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
