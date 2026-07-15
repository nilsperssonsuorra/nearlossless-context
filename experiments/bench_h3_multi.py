"""
H3 multi-needle kill: does critical±R* + scorers still work with 2–3 secrets?

Needles planted at different depths. Tasks:
  recall_all  — answer must contain all codes + passwords (ε=0)
  recall_one  — ask only for the mid needle (selective retrieval)

Arms:
  full
  oracle_r1     — sinks ∪ all crit±R ∪ recent
  anti_oracle   — same budget, no critical tokens
  recent
  posthoc@B     — full prefill + seed_valley
  stream@B      — online seed_valley

H3 support if:
  full ok AND oracle_r1 ≈ full AND anti_oracle fails AND
  stream/posthoc retain multi-span quality (or we learn budget scaling).

Usage:
  python experiments/bench_h3_multi.py
  python experiments/bench_h3_multi.py --ctx 4096 --n-needles 3
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

from bench_h1_oracle import (  # noqa: E402
    build_index_set,
    compress_keep_indices,
    expand_critical,
    find_minimal_span,
    span_recall,
)
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from scorer_valley import compress_with_seed_valley, prefill_streaming_valley  # noqa: E402
from snapkv import cache_nbytes, cache_seq_len, compress_recent, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402

FILLER = (
    "Routine status log: sensors nominal, telemetry within range, no anomalies detected. "
    "Maintenance checklist item completed; continue standard operations as scheduled. "
)

# Distinct multi-needle facts (avoid substring collisions)
NEEDLES = [
    {
        "id": "A",
        "text": (
            "The secret project code is BLUE-ORBIT-7742 and the archive password "
            "is maple-quartz-19."
        ),
        "keys": ["BLUE-ORBIT-7742", "maple-quartz-19"],
    },
    {
        "id": "B",
        "text": (
            "Secondary vault entry: project code CRIMSON-VECTOR-3301 and password "
            "pine-nebula-88."
        ),
        "keys": ["CRIMSON-VECTOR-3301", "pine-nebula-88"],
    },
    {
        "id": "C",
        "text": (
            "Tertiary dossier lists project code SILVER-LATTICE-901 and password "
            "oak-cipher-42."
        ),
        "keys": ["SILVER-LATTICE-901", "oak-cipher-42"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H3 multi-needle kill experiment")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--n-needles", type=int, default=3, choices=[2, 3])
    p.add_argument(
        "--tasks",
        default="recall_all,recall_one",
        help="recall_all and/or recall_one",
    )
    p.add_argument(
        "--methods",
        default="full,oracle_r1,anti_oracle,recent,posthoc,posthoc_mp,stream,stream_mp",
    )
    p.add_argument("--budget", type=int, default=512, help="posthoc/stream budget")
    p.add_argument("--stream-budget", type=int, default=0, help="0 → use --budget")
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--sink-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--min-sep", type=int, default=64, help="multipeak seed separation")
    p.add_argument("--max-peaks", type=int, default=12)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return p.parse_args()


def _has_key(text: str, k: str) -> bool:
    pat = re.escape(k).replace(r"\-", r"[-_]?")
    return re.search(rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])", text, re.I) is not None


def score_multi(text: str, needles: list[dict], *, which: str = "all") -> dict:
    """
    which=all → all keys of all needles
    which=B → only needle id B keys
    """
    if which == "all":
        keys = [k for n in needles for k in n["keys"]]
    else:
        keys = []
        for n in needles:
            if n["id"] == which:
                keys = list(n["keys"])
                break
    hits = {k: _has_key(text, k) for k in keys}
    n_ok = sum(1 for v in hits.values() if v)
    return {
        "success": n_ok == len(keys) and len(keys) > 0,
        "hits": n_ok,
        "n_keys": len(keys),
        "hit_map": hits,
        "answer": text[:400].replace("\n", " "),
    }


def build_multi_prompt(
    tokenizer,
    target_tokens: int,
    needles: list[dict],
    *,
    task: str,
) -> tuple[str, list[float]]:
    """
    Plant needles at evenly spaced depths in the haystack.
    Returns (prompt_text, depths_used).
    """
    n = len(needles)
    depths = [(i + 0.5) / n for i in range(n)]  # centers of n bins
    # For recall_one, still plant all; question targets mid needle only
    body_budget = max(target_tokens - 160, 256)
    chunks: list[str] = []
    while True:
        text_try = "".join(chunks)
        if len(tokenizer.encode(text_try, add_special_tokens=False)) >= body_budget:
            break
        chunks.append(FILLER)
    hay = "".join(chunks)

    # Insert from end to start so earlier positions stay valid
    inserts = []
    for needle, d in zip(needles, depths):
        pos = int(d * max(len(hay) - 1, 0))
        pos = hay.rfind(" ", 0, pos + 1)
        if pos < 0:
            pos = 0
        inserts.append((pos, needle["text"]))
    inserts.sort(key=lambda x: x[0], reverse=True)
    body = hay
    for pos, text in inserts:
        body = body[:pos] + " " + text + " " + body[pos:]

    if task == "recall_all":
        q = (
            "List every secret project code and archive password mentioned in the context. "
            "Answer with the exact codes and passwords only."
        )
    elif task == "recall_one":
        mid = needles[n // 2]
        q = (
            f"There are multiple secrets in the context. "
            f"What is the project code whose password is {mid['keys'][1]}? "
            f"Also report that password. Answer with the exact code and password only."
        )
    else:
        raise ValueError(task)

    user = f"Context:\n{body}\n\nQuestion: {q}"
    messages = [{"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user + "\nAssistant:"

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        keep_head = target_tokens // 4
        keep_tail = target_tokens - keep_head
        ids = ids[:keep_head] + ids[-keep_tail:]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, depths


def find_all_keys_spans(tokenizer, input_ids: torch.Tensor, needles: list[dict]) -> list[int]:
    ids = input_ids[0].tolist()
    spans: set[int] = set()
    for n in needles:
        for key in n["keys"]:
            for t in find_minimal_span(tokenizer, ids, key):
                spans.add(t)
    return sorted(spans)


@torch.inference_mode()
def run_arm(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    method: str,
    budget: int,
    stream_budget: int,
    window: int,
    sinks: int,
    R: int,
    chunk_size: int,
    max_new: int,
    critical: list[int],
    needles: list[dict],
    task: str,
    min_sep: int = 64,
    max_peaks: int = 12,
) -> dict:
    seq_len = int(input_ids.shape[-1])
    keep: list[int] | None = None
    which = "all" if task == "recall_all" else needles[len(needles) // 2]["id"]

    try:
        if method == "full":
            past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
            keep = list(range(seq_len))
        elif method == "oracle_r1":
            past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
            keep = build_index_set(
                seq_len,
                sinks=sinks,
                recent=window,
                critical=critical,
                mode="oracle_ctx",
                span_context=R,
            )
            past = compress_keep_indices(past, keep)
        elif method == "anti_oracle":
            past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
            keep = build_index_set(
                seq_len,
                sinks=sinks,
                recent=window,
                critical=critical,
                mode="anti_oracle",
                span_context=R,
            )
            # match oracle size
            ora = build_index_set(
                seq_len,
                sinks=sinks,
                recent=window,
                critical=critical,
                mode="oracle_ctx",
                span_context=R,
            )
            # rebuild anti with oracle_ctx budget
            keep = build_index_set(
                seq_len,
                sinks=sinks,
                recent=window,
                critical=critical,
                mode="anti_oracle",
                span_context=R,
            )
            # anti_oracle in h1 uses bare oracle size not oracle_ctx — pad/trim
            if len(keep) < len(ora):
                crit_set = set(critical)
                for i in range(seq_len):
                    if i not in keep and i not in crit_set:
                        keep.append(i)
                    if len(keep) >= len(ora):
                        break
                keep = sorted(keep)[: len(ora)]
            elif len(keep) > len(ora):
                keep = keep[: len(ora)]
            past = compress_keep_indices(past, keep)
        elif method == "recent":
            past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
            past = compress_recent(past, budget)
            keep = list(range(max(0, seq_len - budget), seq_len))
        elif method in ("posthoc", "posthoc_mp"):
            past, logits = prefill_chunked(model, input_ids, chunk_size=chunk_size)
            past, keep = compress_with_seed_valley(
                model,
                input_ids,
                past,
                budget=budget,
                window_size=window,
                sinks=sinks,
                expand_radius=R,
                mode="multipeak" if method == "posthoc_mp" else "valley",
                min_sep=min_sep,
                max_peaks=max_peaks,
            )
        elif method in ("stream", "stream_mp"):
            past, logits, st = prefill_streaming_valley(
                model,
                input_ids,
                stream_budget=stream_budget,
                final_budget=stream_budget,
                chunk_size=chunk_size,
                window_size=window,
                sinks=sinks,
                expand_radius=R,
                mode="multipeak" if method == "stream_mp" else "valley",
                min_sep=min_sep,
                max_peaks=max_peaks,
            )
            keep = None
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
        sc = score_multi(
            tokenizer.decode(toks, skip_special_tokens=True),
            needles,
            which=which,
        )
        return {
            "status": "ok",
            "success": sc["success"],
            "hits": sc["hits"],
            "n_keys": sc["n_keys"],
            "span_recall_crit": span_recall(keep, critical) if keep is not None else None,
            "keep_count": len(keep) if keep is not None else cache_seq_len(past),
            "cache_tokens": cache_seq_len(past),
            "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
            "answer": sc["answer"][:200],
            "which": which,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "success": False,
            "hits": 0,
            "n_keys": 0,
            "span_recall_crit": None,
            "keep_count": None,
            "cache_tokens": None,
            "kv_mb": None,
            "answer": "",
            "which": which,
        }
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "status": f"ERROR:{type(e).__name__}",
            "success": False,
            "hits": 0,
            "n_keys": 0,
            "span_recall_crit": None,
            "keep_count": None,
            "cache_tokens": None,
            "kv_mb": None,
            "answer": str(e)[:200],
            "which": which,
        }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    needles = NEEDLES[: args.n_needles]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    stream_budget = args.stream_budget or args.budget
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== H3 multi-needle ===", flush=True)
    print(
        f"n_needles={len(needles)} ctx={args.ctx} budget={args.budget} "
        f"stream_budget={stream_budget} R*={args.R}",
        flush=True,
    )
    print(f"tasks={tasks} methods={methods}", flush=True)

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

    rows: list[dict] = []
    for task in tasks:
        prompt, depths = build_multi_prompt(
            tokenizer, args.ctx, needles, task=task
        )
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])
        critical = find_all_keys_spans(tokenizer, input_ids, needles)
        critical_r = expand_critical(critical, seq_len, args.R)
        print(
            f"\n=== task={task} seq={seq_len} n_crit={len(critical)} "
            f"n_crit±R={len(critical_r)} depths≈{[round(d,2) for d in depths]} ===",
            flush=True,
        )
        if len(critical) < 2:
            print("  WARNING: few critical spans found", flush=True)

        for method in methods:
            print(f"  {method}…", flush=True)
            row = run_arm(
                model,
                tokenizer,
                input_ids,
                method=method,
                budget=args.budget,
                stream_budget=stream_budget,
                window=args.window,
                sinks=args.sink_size,
                R=args.R,
                chunk_size=args.chunk_size,
                max_new=args.max_new,
                critical=critical,
                needles=needles,
                task=task,
                min_sep=args.min_sep,
                max_peaks=args.max_peaks,
            )
            scored = method in (
                "posthoc",
                "posthoc_mp",
                "stream",
                "stream_mp",
                "recent",
            )
            row.update(
                {
                    "model": args.model,
                    "task": task,
                    "method": method,
                    "budget": args.budget if scored else -1,
                    "stream_budget": (
                        stream_budget if method in ("stream", "stream_mp") else None
                    ),
                    "ctx_actual": seq_len,
                    "n_needles": len(needles),
                    "n_critical": len(critical),
                    "R": args.R,
                    "min_sep": args.min_sep if "mp" in method else None,
                }
            )
            rows.append(row)
            rc = row.get("span_recall_crit")
            rc_s = f"{rc:.2f}" if isinstance(rc, float) else "n/a"
            print(
                f"    status={row['status']} success={row['success']} "
                f"hits={row['hits']}/{row['n_keys']} recall_c={rc_s} "
                f"cache={row['cache_tokens']} "
                f"ans={row.get('answer', '')[:60]!r}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary + verdict
    print("\n=== Summary ===", flush=True)
    summary: dict = {}
    for task in tasks:
        summary[task] = {}
        for method in methods:
            sub = [
                r
                for r in rows
                if r.get("task") == task
                and r.get("method") == method
                and r.get("status") == "ok"
            ]
            if not sub:
                continue
            ok = sum(1 for r in sub if r["success"])
            summary[task][method] = {
                "ok": ok,
                "n": len(sub),
                "rate": ok / len(sub),
            }
            print(
                f"  {task:12s} {method:12s}  {100*ok/len(sub):.0f}% "
                f"({ok}/{len(sub)})",
                flush=True,
            )

    def rate(task: str, method: str) -> float | None:
        s = summary.get(task, {}).get(method)
        return s["rate"] if s else None

    # Primary task for verdict: recall_all
    full = rate("recall_all", "full")
    ora = rate("recall_all", "oracle_r1")
    anti = rate("recall_all", "anti_oracle")
    stream = rate("recall_all", "stream")
    post = rate("recall_all", "posthoc")
    recent = rate("recall_all", "recent")

    verdict = "INCONCLUSIVE"
    reasons = []
    if full is not None and full < 1.0:
        verdict = "SUITE_INVALID"
        reasons.append("full failed multi-needle recall_all — fix suite/prompt")
    elif full == 1.0 and ora == 1.0 and (anti is not None and anti == 0.0):
        if (stream == 1.0) or (post == 1.0):
            verdict = "H3_SUPPORTED"
            reasons.append(
                "oracle±R* matches full on multi-needle; anti-oracle fails; "
                "non-oracle scorer recovers all secrets at budget"
            )
        elif stream == 0.0 and post == 0.0:
            verdict = "H3_ORACLE_ONLY"
            reasons.append(
                "multi-needle needs all critical±R* (oracle ok, anti fails) "
                "but scorers miss some spans at this budget — raise budget"
            )
        else:
            verdict = "H3_PARTIAL"
            reasons.append("oracle path holds; scorers mixed")
        if recent == 0.0:
            reasons.append("recent-only fails (needles not only at end)")
    elif full == 1.0 and ora is not None and ora < 1.0:
        verdict = "H3_NEEDS_MORE_CONTEXT"
        reasons.append("oracle_r1 insufficient for multi-needle — need larger R or more structure")
    elif full == 1.0 and anti is not None and anti > 0.0:
        verdict = "H3_NECESSITY_WEAK"
        reasons.append("anti-oracle sometimes works — leakage or weak multi-needle")
    else:
        reasons.append("see summary rates")

    # recall_one note
    one_full = rate("recall_one", "full")
    if one_full is not None:
        reasons.append(f"recall_one full rate={one_full}")

    print(f"\nVERDICT: {verdict}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h3_multi_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "n_needles": len(needles),
        "budget": args.budget,
        "stream_budget": stream_budget,
        "R": args.R,
        "ctx": args.ctx,
        "csv": str(out),
        "model": args.model,
        "hypothesis": "H3 multi-needle still requires critical±R* retention for all facts",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
