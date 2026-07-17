"""
Stress-test query-unknown novelty detector beyond structured code needles.

Scenarios (multi-seed × depths):
  code   — original BLUE-ORBIT / maple-quartz (control)
  nl     — natural-language fact (names/places, few digits)
  adv    — code needle but filler is full of fake ID-like strings
  multi3 — 3 structured needles, recall_all

Arms @ stream budgets:
  valley, novelty, oracle_pin

Usage:
  python experiments/bench_novelty_stress.py
  python experiments/bench_novelty_stress.py --scenarios code,nl,adv --budgets 512
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_h1_oracle import find_minimal_span  # noqa: E402
from bench_h3_multi import NEEDLES as MULTI_NEEDLES  # noqa: E402
from bench_needle import FILLER, FILLER_VARIANTS, QUESTION  # noqa: E402
from capsules import prefill_streaming_oracle_pin  # noqa: E402
from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from novelty_detect import prefill_streaming_novelty_pin  # noqa: E402
from scorer_valley import prefill_streaming_valley  # noqa: E402
from snapkv import cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402

# --- scenario payloads ---
CODE_NEEDLE = (
    "The secret project code is BLUE-ORBIT-7742 and the archive password is maple-quartz-19."
)
CODE_KEYS = ["BLUE-ORBIT-7742", "maple-quartz-19"]
CODE_Q = QUESTION

NL_NEEDLE = (
    "Personnel note: the ambassador's middle name is Seraphine and she was born in Reykjavik."
)
NL_KEYS = ["Seraphine", "Reykjavik"]
NL_Q = (
    "What is the ambassador's middle name and city of birth mentioned in the context? "
    "Answer with the exact name and city only."
)

# Adversarial filler: looks ID-like / code-like so novelty is flooded
ADV_FILLER_UNITS = [
    "Telemetry tag CODE-ALPHA-1200 status nominal; checksum ZETA-991-beta verified. ",
    "Packet id PACKET-77-GAMMA routed; auth token TOKEN-X9Y2-DELTA accepted. ",
    "Sensor REF-8801-Q reported OK; spare part SPARE-44-OMEGA logged. ",
    "Session key KEY-AB12-CD34 rotated; link PROFILE-9-SIGMA stable. ",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Novelty detector stress suite")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--depths", default="0.0,0.5,1.0")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--budgets", default="512")
    p.add_argument(
        "--scenarios",
        default="code,nl,adv,multi3,hop2",
        help="Comma list: code,nl,adv,multi3,hop2",
    )
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--R", type=int, default=1)
    p.add_argument(
        "--max-capsules",
        type=int,
        default=12,
        help="Novelty max capsules (raise for multi-entity)",
    )
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--arms",
        default="valley,novelty,oracle_pin",
        help="Comma list: valley,novelty,oracle_pin,full",
    )
    return p.parse_args()


def _grow_body(tokenizer, body_budget: int, unit_fn, seed: int | None) -> str:
    rng = random.Random(seed) if seed is not None else None
    chunks: list[str] = []
    while True:
        text = "".join(chunks)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n >= body_budget:
            break
        chunks.append(unit_fn(rng))
    return "".join(chunks)


def build_scenario_prompt(
    tokenizer,
    *,
    scenario: str,
    target_tokens: int,
    depth: float,
    seed: int | None = None,
) -> tuple[str, list[str], str]:
    """Return (prompt_text, keys, question)."""
    depth = max(0.0, min(1.0, depth))
    body_budget = max(target_tokens - 160, 256)
    rng = random.Random(seed) if seed is not None else None

    if scenario == "code":
        needle, keys, question = CODE_NEEDLE, CODE_KEYS, CODE_Q

        def unit(r):
            if r is None:
                return FILLER
            return r.choice(FILLER_VARIANTS)

    elif scenario == "nl":
        needle, keys, question = NL_NEEDLE, NL_KEYS, NL_Q

        def unit(r):
            if r is None:
                return FILLER
            return r.choice(FILLER_VARIANTS)

    elif scenario == "adv":
        needle, keys, question = CODE_NEEDLE, CODE_KEYS, CODE_Q

        def unit(r):
            if r is None:
                return ADV_FILLER_UNITS[0]
            return r.choice(ADV_FILLER_UNITS)

    elif scenario == "multi3":
        keys = []
        for nd in MULTI_NEEDLES[:3]:
            keys.extend(nd["keys"])
        question = (
            "List every secret project code and archive password mentioned in the context. "
            "Answer with the exact codes and passwords only."
        )
        # plant three needles at 0.15, 0.5, 0.85 after building hay
        def unit(r):
            if r is None:
                return FILLER
            return r.choice(FILLER_VARIANTS)

        hay = _grow_body(tokenizer, body_budget, unit, seed)
        # place three needles
        fracs = [0.15, 0.50, 0.85]
        body = hay
        # insert from end so earlier positions stay stable
        for frac, nd in zip(reversed(fracs), reversed(MULTI_NEEDLES[:3])):
            pos = int(frac * max(len(body) - 1, 0))
            pos = body.rfind(" ", 0, pos + 1)
            if pos < 0:
                pos = 0
            body = body[:pos] + " " + nd["text"] + " " + body[pos:]
        user = f"Context:\n{body}\n\nQuestion: {question}"
        return _chat_wrap(tokenizer, user, target_tokens), keys, question

    elif scenario == "hop2":
        # Two linked facts; answer needs both neighborhoods
        keys = ["maple-quartz-19"]  # primary success key
        # Also track link id for partial diagnostics (not required for success)
        question = (
            "What is Alice's vault password? "
            "Use the employee identifier to link records. Answer with the exact password only."
        )
        fact1 = "Personnel note: Alice has employee identifier E-4412 in the registry."
        fact2 = "Security note: the vault password for employee E-4412 is maple-quartz-19."

        def unit(r):
            if r is None:
                return FILLER
            return r.choice(FILLER_VARIANTS)

        hay = _grow_body(tokenizer, body_budget, unit, seed)
        p1 = int(0.3 * max(len(hay) - 1, 0))
        p2 = int(0.7 * max(len(hay) - 1, 0))
        if rng is not None and len(hay) > 64:
            p1 = max(0, min(len(hay) - 1, p1 + rng.randint(-24, 24)))
            p2 = max(0, min(len(hay) - 1, p2 + rng.randint(-24, 24)))
            if p1 > p2:
                p1, p2 = p2, p1
        p1 = hay.rfind(" ", 0, p1 + 1)
        p2 = hay.rfind(" ", 0, p2 + 1)
        if p1 < 0:
            p1 = 0
        if p2 < 0:
            p2 = 0
        body = hay[:p2] + " " + fact2 + " " + hay[p2:]
        body = body[:p1] + " " + fact1 + " " + body[p1:]
        user = f"Context:\n{body}\n\nQuestion: {question}"
        return _chat_wrap(tokenizer, user, target_tokens), keys, question

    else:
        raise ValueError(scenario)

    hay = _grow_body(tokenizer, body_budget, unit, seed)
    pos = int(depth * max(len(hay) - 1, 0))
    if rng is not None and len(hay) > 64:
        pos = max(0, min(len(hay) - 1, pos + rng.randint(-32, 32)))
    pos = hay.rfind(" ", 0, pos + 1)
    if pos < 0:
        pos = 0
    body = hay[:pos] + " " + needle + " " + hay[pos:]
    user = f"Context:\n{body}\n\nQuestion: {question}"
    return _chat_wrap(tokenizer, user, target_tokens), keys, question


def _chat_wrap(tokenizer, user: str, target_tokens: int) -> str:
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
    return text


def find_critical(tokenizer, input_ids: torch.Tensor, keys: list[str]) -> list[int]:
    ids = input_ids[0].tolist()
    spans: set[int] = set()
    for key in keys:
        for t in find_minimal_span(tokenizer, ids, key):
            spans.add(t)
    return sorted(spans)


def score_keys(text: str, keys: list[str]) -> dict:
    def has_key(k: str) -> bool:
        pat = re.escape(k).replace(r"\-", r"[-_]?")
        return re.search(rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])", text, re.I) is not None

    hits = {k: has_key(k) for k in keys}
    n = sum(1 for v in hits.values() if v)
    return {
        "hits": n,
        "n_keys": len(keys),
        "success": n == len(keys),
        "answer": text[:400],
        "hit_map": hits,
    }


@torch.inference_mode()
def decode(model, tokenizer, past, logits, seq_len, max_new):
    return greedy_generate(
        model,
        past,
        logits,
        max_new,
        eos_id=tokenizer.eos_token_id,
        next_position=seq_len,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== Novelty stress suite ===", flush=True)
    print(
        f"scenarios={scenarios} seeds={seeds} depths={depths} budgets={budgets} arms={arms}",
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

    for scenario in scenarios:
        for seed in seeds:
            # multi3 / hop2 use fixed plant patterns (depth loop unused)
            depth_list = [0.5] if scenario in ("multi3", "hop2") else depths
            for depth in depth_list:
                prompt, keys, _q = build_scenario_prompt(
                    tokenizer,
                    scenario=scenario,
                    target_tokens=args.ctx,
                    depth=depth,
                    seed=seed,
                )
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    device
                )
                seq_len = int(input_ids.shape[-1])
                critical = find_critical(tokenizer, input_ids, keys)
                print(
                    f"\n--- {scenario} seed={seed} depth={depth} seq={seq_len} "
                    f"n_crit={len(critical)} keys={keys[:4]} ---",
                    flush=True,
                )
                if not critical:
                    print("  WARN: no critical spans found", flush=True)

                for B in budgets:
                    for arm in arms:
                        try:
                            if arm == "full":
                                past, logits = prefill_chunked(
                                    model, input_ids, chunk_size=512
                                )
                                st = {"peak_cache": seq_len}
                            elif arm == "valley":
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
                            elif arm == "novelty":
                                past, logits, st = prefill_streaming_novelty_pin(
                                    model,
                                    tokenizer,
                                    input_ids,
                                    stream_budget=B,
                                    final_budget=B,
                                    chunk_size=512,
                                    window_size=args.window,
                                    sinks=8,
                                    expand_radius=args.R,
                                    max_capsules=args.max_capsules,
                                )
                            elif arm == "oracle_pin":
                                past, logits, st = prefill_streaming_oracle_pin(
                                    model,
                                    input_ids,
                                    critical=critical,
                                    stream_budget=B,
                                    final_budget=B,
                                    chunk_size=512,
                                    window_size=args.window,
                                    sinks=8,
                                    expand_radius=args.R,
                                )
                            else:
                                raise ValueError(arm)

                            toks = decode(
                                model, tokenizer, past, logits, seq_len, args.max_new
                            )
                            sc = score_keys(
                                tokenizer.decode(toks, skip_special_tokens=True), keys
                            )
                            row = {
                                "scenario": scenario,
                                "arm": f"{arm}@{B}" if arm != "full" else "full",
                                "seed": seed,
                                "depth": depth,
                                "budget": B if arm != "full" else seq_len,
                                "success": sc["success"],
                                "hits": sc["hits"],
                                "n_keys": sc["n_keys"],
                                "cache_tokens": cache_seq_len(past),
                                "peak_cache": st.get(
                                    "peak_cache", st.get("peak_cache_tokens")
                                ),
                                "n_crit": len(critical),
                                "answer": sc["answer"][:100].replace("\n", " "),
                            }
                            rows.append(row)
                            print(
                                f"  {row['arm']}: ok={sc['success']} hits={sc['hits']}/{sc['n_keys']}",
                                flush=True,
                            )
                            del past
                        except Exception as e:
                            rows.append(
                                {
                                    "scenario": scenario,
                                    "arm": f"{arm}@{B}",
                                    "seed": seed,
                                    "depth": depth,
                                    "success": False,
                                    "status": f"ERR:{type(e).__name__}",
                                    "answer": str(e)[:120],
                                }
                            )
                            print(f"  {arm}@{B}: ERR {e}", flush=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # Summary table
    print("\n=== Summary (success rate) ===", flush=True)
    summary: dict[str, dict] = {}
    for scenario in scenarios:
        summary[scenario] = {}
        for arm in arms:
            for B in budgets:
                key = "full" if arm == "full" else f"{arm}@{B}"
                sel = [
                    r
                    for r in rows
                    if r.get("scenario") == scenario
                    and r.get("arm") == key
                    and not str(r.get("status", "")).startswith("ERR")
                ]
                if not sel:
                    continue
                ok = sum(1 for r in sel if r.get("success"))
                rate = ok / len(sel)
                summary[scenario][key] = {
                    "n": len(sel),
                    "ok": ok,
                    "rate": round(rate, 4),
                }
                print(f"  {scenario:7} {key:20} {ok}/{len(sel)} = {rate:.3f}", flush=True)

    # Verdicts per scenario
    verdicts = {}
    for scenario in scenarios:
        nov = summary.get(scenario, {}).get("novelty@512") or summary.get(
            scenario, {}
        ).get(f"novelty@{budgets[0]}")
        val = summary.get(scenario, {}).get("valley@512") or summary.get(
            scenario, {}
        ).get(f"valley@{budgets[0]}")
        ora = summary.get(scenario, {}).get("oracle_pin@512") or summary.get(
            scenario, {}
        ).get(f"oracle_pin@{budgets[0]}")
        if not nov:
            verdicts[scenario] = "NO_DATA"
            continue
        nr, vr = nov["rate"], (val or {}).get("rate")
        if nr >= 0.9 and (vr is None or nr > vr + 0.15):
            verdicts[scenario] = "NOVELTY_ROBUST"
        elif nr >= 0.9:
            verdicts[scenario] = "NOVELTY_OK"
        elif vr is not None and nr > vr + 0.15:
            verdicts[scenario] = "NOVELTY_HELPS"
        elif vr is not None and nr < vr - 0.1:
            verdicts[scenario] = "NOVELTY_REGRESS"
        else:
            verdicts[scenario] = "NOVELTY_WEAK"
        if ora and ora["rate"] >= 0.9 and nr < 0.7:
            verdicts[scenario] = "STILL_DISCOVERY_GAP"

    print("\nVERDICTS:", verdicts, flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"novelty_stress_{stamp}.csv"
    write_csv(out, rows)
    meta = {
        "verdicts": verdicts,
        "summary": summary,
        "model": args.model,
        "scenarios": scenarios,
        "seeds": seeds,
        "depths": depths,
        "budgets": budgets,
        "csv": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
