"""
Harder multi-hop: 3-link chain + distractors.

Chain:
  Alice works in Department-7.
  Department-7 vault officer id is E-4412.
  Vault password for employee E-4412 is maple-quartz-19.

Distractors (wrong links):
  Bob → Department-3 → E-9901 → pine-nebula-88
  Carol → Department-7 → E-1100 → oak-cipher-42  (same dept, wrong officer)

Q: What is Alice's vault password? → maple-quartz-19 only

Usage:
  python experiments/bench_h3_hop3.py --ctx 4096
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

from adaptive import policy_for  # noqa: E402
from bench_h1_oracle import (  # noqa: E402
    build_index_set,
    compress_keep_indices,
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

# True chain
F_ALICE = "Assignment: Alice is assigned to Department-7 for the current quarter."
F_DEPT = "Registry: the vault officer for Department-7 has employee identifier E-4412."
F_PASS = "Security: the vault password for employee E-4412 is maple-quartz-19."

# Distractors
F_BOB = "Assignment: Bob is assigned to Department-3 for the current quarter."
F_DEPT3 = "Registry: the vault officer for Department-3 has employee identifier E-9901."
F_BOB_PASS = "Security: the vault password for employee E-9901 is pine-nebula-88."
F_CAROL = "Note: Carol also works near Department-7 facilities."
F_WRONG_OFF = "Registry alternate: Department-7 backup officer is E-1100."
F_WRONG_PASS = "Security: the vault password for employee E-1100 is oak-cipher-42."

ANSWER = "maple-quartz-19"
WRONG = ["pine-nebula-88", "oak-cipher-42"]
TRUE_KEYS = [
    "Alice",
    "Department-7",
    "E-4412",
    "maple-quartz-19",
    "vault officer for Department-7",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-hop multi-hop with distractors")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def _insert(hay: str, depth: float, text: str) -> str:
    pos = int(depth * max(len(hay) - 1, 0))
    pos = max(hay.rfind(" ", 0, pos + 1), 0)
    return hay[:pos] + " " + text + " " + hay[pos:]


def build_prompt(tokenizer, target_tokens: int) -> str:
    body_budget = max(target_tokens - 180, 256)
    chunks: list[str] = []
    while True:
        t = "".join(chunks)
        if len(tokenizer.encode(t, add_special_tokens=False)) >= body_budget:
            break
        chunks.append(FILLER)
    hay = "".join(chunks)
    # depths spread; insert from end so earlier depths stay valid-ish
    facts = [
        (0.85, F_WRONG_PASS),
        (0.78, F_PASS),
        (0.70, F_BOB_PASS),
        (0.62, F_WRONG_OFF),
        (0.55, F_DEPT),
        (0.48, F_DEPT3),
        (0.40, F_CAROL),
        (0.32, F_BOB),
        (0.22, F_ALICE),
    ]
    body = hay
    for d, text in sorted(facts, key=lambda x: -x[0]):
        body = _insert(body, d, text)

    q = (
        "What is Alice's vault password? "
        "Follow only Alice's department → that department's vault officer → "
        "that officer's password. Ignore other employees. "
        "Answer with the exact password only."
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
        kh = target_tokens // 4
        text = tokenizer.decode(
            ids[:kh] + ids[-(target_tokens - kh) :], skip_special_tokens=False
        )
    return text


def score_answer(text: str) -> dict:
    def has(k: str) -> bool:
        pat = re.escape(k).replace(r"\-", r"[-_]?")
        return re.search(rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])", text, re.I) is not None

    ok = has(ANSWER)
    wrong_hit = any(has(w) for w in WRONG)
    return {
        "success": ok and not wrong_hit,
        "has_answer": ok,
        "wrong_distractor": wrong_hit,
        "answer": text[:300].replace("\n", " "),
    }


def find_critical(tokenizer, input_ids: torch.Tensor) -> list[int]:
    ids = input_ids[0].tolist()
    spans: set[int] = set()
    for key in TRUE_KEYS:
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
    pol = policy_for(n_entities=3, L=args.ctx, multi_hop=True)

    print("=== 3-hop + distractors ===", flush=True)
    print(f"adaptive policy: {pol}", flush=True)

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

    prompt = build_prompt(tok, args.ctx)
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = int(input_ids.shape[-1])
    critical = find_critical(tok, input_ids)
    print(f"seq={seq_len} n_crit={len(critical)}", flush=True)

    arms = [
        ("full", None, None),
        ("oracle_r1", None, 1),
        ("anti_oracle", None, 1),
        ("recent", 512, 1),
        ("posthoc_r1_512", 512, 1),
        ("posthoc_r8_384", 384, 8),
        ("posthoc_adaptive", pol.budget, pol.R),
        ("stream_r1_512", 512, 1),
        ("stream_r8_1024", 1024, 8),
        ("stream_adaptive", pol.stream_budget, pol.R),
    ]

    rows = []
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
                past = compress_recent(past, budget or 512)
            elif name.startswith("posthoc"):
                past, logits = prefill_chunked(
                    model, input_ids, chunk_size=args.chunk_size
                )
                past, keep = compress_with_seed_valley(
                    model,
                    input_ids,
                    past,
                    budget=int(budget),
                    window_size=args.window,
                    sinks=8,
                    expand_radius=int(R),
                )
            elif name.startswith("stream"):
                b = int(budget)
                past, logits, _ = prefill_streaming_valley(
                    model,
                    input_ids,
                    stream_budget=b,
                    final_budget=b,
                    chunk_size=args.chunk_size,
                    window_size=args.window,
                    sinks=8,
                    expand_radius=int(R),
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
            sc = score_answer(tok.decode(toks, skip_special_tokens=True))
            row = {
                "arm": name,
                "success": sc["success"],
                "has_answer": sc["has_answer"],
                "wrong_distractor": sc["wrong_distractor"],
                "status": "ok",
                "cache_tokens": cache_seq_len(past),
                "kv_mb": round(cache_nbytes(past) / (1024**2), 3),
                "span_recall": span_recall(keep, critical) if keep else None,
                "answer": sc["answer"][:160],
                "budget": budget,
                "R": R,
            }
            print(
                f"    ok={row['success']} wrong={row['wrong_distractor']} "
                f"cache={row['cache_tokens']} ans={row['answer'][:48]!r}",
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_ok = any(r["arm"] == "full" and r.get("success") for r in rows)
    ora_ok = any(r["arm"] == "oracle_r1" and r.get("success") for r in rows)
    anti_ok = any(r["arm"] == "anti_oracle" and r.get("success") for r in rows)
    scorers = [r for r in rows if r["arm"].startswith(("posthoc", "stream"))]
    if not full_ok:
        verdict = "SUITE_INVALID"
    elif ora_ok and not anti_ok and any(r.get("success") for r in scorers):
        verdict = "HOP3_SUPPORTED"
    elif ora_ok and not anti_ok:
        verdict = "HOP3_ORACLE_ONLY"
    elif ora_ok and anti_ok:
        verdict = "HOP3_NECESSITY_WEAK"
    else:
        verdict = "HOP3_ORACLE_FAIL"

    print(f"\nVERDICT: {verdict}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"h3_hop3_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdict": verdict,
        "policy": pol.__dict__,
        "csv": str(out),
        "answer": ANSWER,
        "hypothesis": "3-hop with distractors still needs true-chain critical spans",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
