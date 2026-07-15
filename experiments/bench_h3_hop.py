"""
Multi-hop smoke: two facts linked by an ID; question needs both spans.

  Fact1 @ depth≈0.3: Alice's employee id is E-4412
  Fact2 @ depth≈0.7: vault password for E-4412 is maple-quartz-19
  Q: What is Alice's vault password? → maple-quartz-19

Arms: full, oracle (both facts ±R), anti (drop both), recent, posthoc R=1/8, stream R=1/8

Usage:
  python experiments/bench_h3_hop.py --ctx 4096
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
FACT1 = "Personnel note: Alice has employee identifier E-4412 in the registry."
FACT2 = "Security note: the vault password for employee E-4412 is maple-quartz-19."
ANSWER_KEY = "maple-quartz-19"
LINK_KEY = "E-4412"
NAME_KEY = "Alice"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-hop two-fact smoke")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def build_hop_prompt(tokenizer, target_tokens: int) -> str:
    body_budget = max(target_tokens - 160, 256)
    chunks: list[str] = []
    while True:
        t = "".join(chunks)
        if len(tokenizer.encode(t, add_special_tokens=False)) >= body_budget:
            break
        chunks.append(FILLER)
    hay = "".join(chunks)
    p1 = int(0.3 * max(len(hay) - 1, 0))
    p2 = int(0.7 * max(len(hay) - 1, 0))
    p1 = max(hay.rfind(" ", 0, p1 + 1), 0)
    p2 = max(hay.rfind(" ", 0, p2 + 1), 0)
    # insert later position first
    body = hay[:p2] + " " + FACT2 + " " + hay[p2:]
    # p1 may shift if p1 < p2 — recompute roughly
    if p1 < p2:
        body = body[:p1] + " " + FACT1 + " " + body[p1:]
    else:
        body = body[:p1] + " " + FACT1 + " " + body[p1:]

    q = (
        "What is Alice's vault password? "
        "Use the employee identifier to link records. Answer with the exact password only."
    )
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
        kh, kt = target_tokens // 4, target_tokens - target_tokens // 4
        text = tokenizer.decode(ids[:kh] + ids[-kt:], skip_special_tokens=False)
    return text


def score_hop(text: str) -> dict:
    pat = re.escape(ANSWER_KEY).replace(r"\-", r"[-_]?")
    ok = re.search(rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])", text, re.I) is not None
    return {"success": ok, "answer": text[:300].replace("\n", " ")}


def find_hop_critical(tokenizer, input_ids: torch.Tensor) -> list[int]:
    ids = input_ids[0].tolist()
    spans: set[int] = set()
    for key in (ANSWER_KEY, LINK_KEY, NAME_KEY, "employee identifier", "vault password"):
        for t in find_minimal_span(tokenizer, ids, key):
            spans.add(t)
    return sorted(spans)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.ctx > 4096:
        args.ctx = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Multi-hop two-fact smoke ===", flush=True)
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

    prompt = build_hop_prompt(tok, args.ctx)
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = int(input_ids.shape[-1])
    critical = find_hop_critical(tok, input_ids)
    print(f"seq={seq_len} n_crit={len(critical)}", flush=True)

    rows = []
    arms = [
        ("full", None, 1),
        ("oracle_r1", None, 1),
        ("anti_oracle", None, 1),
        ("recent", args.budget, 1),
        ("posthoc_r1", args.budget, 1),
        ("posthoc_r8", args.budget, 8),
        ("stream_r1", args.budget, 1),
        ("stream_r8", args.budget, 8),
        ("posthoc_r8_384", 384, 8),
        ("stream_r8_1024", 1024, 8),
    ]

    for name, budget, R in arms:
        print(f"  {name}…", flush=True)
        try:
            keep = None
            if name == "full":
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                keep = list(range(seq_len))
            elif name == "oracle_r1":
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                keep = build_index_set(
                    seq_len,
                    sinks=8,
                    recent=args.window,
                    critical=critical,
                    mode="oracle_ctx",
                    span_context=1,
                )
                past = compress_keep_indices(past, keep)
            elif name == "anti_oracle":
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                keep = build_index_set(
                    seq_len,
                    sinks=8,
                    recent=args.window,
                    critical=critical,
                    mode="anti_oracle",
                    span_context=1,
                )
                past = compress_keep_indices(past, keep)
            elif name == "recent":
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                past = compress_recent(past, budget or args.budget)
            elif name.startswith("posthoc"):
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                past, keep = compress_with_seed_valley(
                    model,
                    input_ids,
                    past,
                    budget=budget or args.budget,
                    window_size=args.window,
                    sinks=8,
                    expand_radius=R,
                )
            elif name.startswith("stream"):
                b = budget or args.budget
                past, logits, _ = prefill_streaming_valley(
                    model,
                    input_ids,
                    stream_budget=b,
                    final_budget=b,
                    chunk_size=args.chunk_size,
                    window_size=args.window,
                    sinks=8,
                    expand_radius=R,
                )
            else:
                raise ValueError(name)

            toks = greedy_generate(
                model,
                past,
                logits,
                args.max_new,
                eos_id=tok.eos_token_id,
                next_position=seq_len,
            )
            sc = score_hop(tok.decode(toks, skip_special_tokens=True))
            row = {
                "arm": name,
                "success": sc["success"],
                "status": "ok",
                "cache_tokens": cache_seq_len(past),
                "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                "span_recall": span_recall(keep, critical) if keep else None,
                "answer": sc["answer"][:160],
                "budget": budget,
                "R": R,
            }
            print(
                f"    ok={row['success']} cache={row['cache_tokens']} "
                f"ans={row['answer'][:50]!r}",
                flush=True,
            )
            del past, logits, toks
        except Exception as e:
            row = {
                "arm": name,
                "success": False,
                "status": f"ERR:{e}",
                "answer": str(e)[:120],
            }
            print(f"    ERROR {e}", flush=True)
            torch.cuda.empty_cache()
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_ok = any(r["arm"] == "full" and r.get("success") for r in rows)
    ora_ok = any(r["arm"] == "oracle_r1" and r.get("success") for r in rows)
    anti_ok = any(r["arm"] == "anti_oracle" and r.get("success") for r in rows)
    if not full_ok:
        verdict = "SUITE_INVALID"
    elif ora_ok and not anti_ok:
        # check if any compressed scorer works
        scorers = [r for r in rows if r["arm"].startswith(("posthoc", "stream"))]
        if any(r.get("success") for r in scorers):
            verdict = "HOP_SUPPORTED"
        else:
            verdict = "HOP_ORACLE_ONLY"
    elif ora_ok and anti_ok:
        verdict = "HOP_NECESSITY_WEAK"
    else:
        verdict = "HOP_ORACLE_FAIL"

    print(f"\nVERDICT: {verdict}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h3_hop_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "csv": str(out),
        "hypothesis": "Multi-hop needs both linked critical spans retained",
        "answer_key": ANSWER_KEY,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
