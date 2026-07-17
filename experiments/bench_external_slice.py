"""
Small external-style eval slice (not full LongBench leaderboard).

Loads a public HF split when available; otherwise falls back to a bundled
offline multi-doc / long-prose set so the bench is always runnable.

Arms: full, stream_valley@B, stream_novelty@B
Metric: token-level F1 vs reference answer(s) + exact substring hit.

Usage:
  python experiments/bench_external_slice.py
  python experiments/bench_external_slice.py --source longbench --n 8 --budget 512
  python experiments/bench_external_slice.py --source offline --n 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PRIMARY_MODEL_ID, RESULTS_DIR, SNAPKV_WINDOW  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from novelty_detect import (  # noqa: E402
    prefill_streaming_hybrid_pin,
    prefill_streaming_novelty_pin,
    prefill_streaming_query_hold,
)
from scorer_valley import prefill_streaming_valley  # noqa: E402
from snapkv import cache_seq_len, prefill_chunked  # noqa: E402
from utils import write_csv  # noqa: E402


# --- offline fallback items (always available) ---
OFFLINE_ITEMS = [
    {
        "id": "off_wiki_q1",
        "context": (
            "The city of Uppsala is home to Scandinavia's oldest university, founded in 1477. "
            "Carl Linnaeus served as a professor of medicine there and developed a system for "
            "naming organisms that is still used worldwide. The university library, Carolina Rediviva, "
            "holds the Codex Argenteus, a 6th-century silver Bible manuscript. "
            "Tourism boards often highlight the botanical garden established by Linnaeus, which "
            "still cultivates species he once classified. Nearby cathedral towers dominate the "
            "skyline, and student nations organize much of campus social life. "
            "Local industry historically included brickmaking and book printing. Modern research "
            "clusters focus on life sciences and materials. Winters are long; summers bring "
            "extended daylight that residents celebrate with outdoor festivals along the Fyris river."
        ),
        "question": "Which manuscript is held at Carolina Rediviva?",
        "answers": ["Codex Argenteus", "silver Bible"],
    },
    {
        "id": "off_tech_q1",
        "context": (
            "Project Orion-Delta was a short-lived attempt to build a radio array on a remote "
            "island. Engineers installed twelve dish antennas in a hexagonal lattice. Power came "
            "from a diesel generator codenamed Iron Finch. Communications with the mainland used "
            "a microwave link at 11 GHz. After three months, salt corrosion forced a shutdown. "
            "The final status report recommended relocating equipment inland before any restart. "
            "Budget notes list spare waveguides and a single spare LNB for emergency swaps. "
            "No scientific papers were published; data tapes were archived under box label D-19."
        ),
        "question": "What was the diesel generator codename?",
        "answers": ["Iron Finch"],
    },
    {
        "id": "off_legal_q1",
        "context": (
            "In the case of Northbridge v. Harbor Cooperative, the court examined whether a "
            "shared dock license could be revoked without written notice. Judge Elena Voss held "
            "that verbal warnings were insufficient under the cooperative bylaws, section 4.2. "
            "The defendant argued custom practice allowed oral notice, but the panel disagreed. "
            "Damages were limited to lost berthing fees for one season. Costs were split equally. "
            "Later commentary praised the clarity of Voss's opinion on notice formalities."
        ),
        "question": "Which judge held that verbal warnings were insufficient?",
        "answers": ["Elena Voss", "Judge Elena Voss", "Voss"],
    },
    {
        "id": "off_med_q1",
        "context": (
            "During the 2019 river fever outbreak, clinics tracked cases using form RF-17. "
            "The index patient was a barge worker who reported symptoms after docking at Pier 9. "
            "Public health officers closed the riverside market for ten days. The causative agent "
            "was later identified as a waterborne bacterium, not a virus as first suspected. "
            "Recovery was typically complete within two weeks with standard antibiotics. "
            "No vaccine was available at the time of the report."
        ),
        "question": "Where did the index patient dock?",
        "answers": ["Pier 9"],
    },
    {
        "id": "off_multi_q1",
        "context": (
            "### Document: Warehouse A\n"
            "Pallets of dry goods stored on level 2. No hazardous materials present.\n\n"
            "### Document: Warehouse B\n"
            "Cold chain package LOT-MIRROR-308 is staged for transfer to Clinic East by Friday.\n\n"
            "### Document: Warehouse C\n"
            "Empty crates returned from retail partners; awaiting recycling pickup.\n"
        ),
        "question": "Which package is staged in Warehouse B, and where is it going?",
        "answers": ["LOT-MIRROR-308", "Clinic East"],
    },
    {
        "id": "off_hist_q1",
        "context": (
            "The Treaty of Glass Harbor (fictional exercise text) ended a naval standoff between "
            "the twin ports of Sable and Merrow. Article III required both sides to demilitarize "
            "the strait within sixty days. Article VII created a joint fisheries board seated in "
            "the neutral town of Greyfen. Ratification was completed on 12 March after merchants "
            "from both ports petitioned for open trade lanes. Historians note that Greyfen later "
            "became a regional banking center as a side effect of the treaty institutions."
        ),
        "question": "Where was the joint fisheries board seated?",
        "answers": ["Greyfen"],
    },
    {
        "id": "off_sci_q1",
        "context": (
            "In a materials study, alloy sample KX-41 showed unexpected ductility after a two-step "
            "anneal at 720 C then 480 C. Researchers attributed the effect to fine carbide "
            "precipitates observed under electron microscopy. Control samples without the second "
            "anneal remained brittle. The team recommended KX-41 for prototype spring components "
            "but not for high-temperature housings."
        ),
        "question": "Which alloy sample showed unexpected ductility after the two-step anneal?",
        "answers": ["KX-41"],
    },
    {
        "id": "off_ops_q1",
        "context": (
            "Shift handover notes for Platform Helios: morning crew reported a pressure drop in "
            "line 4B. Afternoon crew replaced a valve gasket and restored nominal flow. Evening "
            "crew logged residual vibration on pump P-17 and requested a bearing inspection at "
            "next planned maintenance. Safety officer approved continued operation at reduced RPM."
        ),
        "question": "Which pump showed residual vibration?",
        "answers": ["P-17", "pump P-17"],
    },
    {
        "id": "off_culture_q1",
        "context": (
            "The traveling exhibit 'Threads of the North' featured textiles from three coastal "
            "villages. The centerpiece was a blue wool cloak embroidered with herring motifs, "
            "loaned by the private collector Mara Elling. Attendance peaked on weekend evenings. "
            "A short film about dye plants played on a loop near the exit."
        ),
        "question": "Who loaned the blue wool cloak?",
        "answers": ["Mara Elling"],
    },
    {
        "id": "off_finance_q1",
        "context": (
            "Quarterly notes for Fund Cedar: equity sleeve returned 3.1% while the bond sleeve "
            "returned 0.4%. The largest single holding remained Northpeak Utilities. Management "
            "reduced exposure to short-duration credit after volatility in March. Cash rose to "
            "8% of net assets awaiting better entry points in industrial names."
        ),
        "question": "What was the largest single holding of Fund Cedar?",
        "answers": ["Northpeak Utilities"],
    },
    {
        "id": "off_geo_q1",
        "context": (
            "Surveyors mapping the eastern ridge found a sinkhole cluster near marker stone 22. "
            "They recommended fencing the area and rerouting the hiking path by 400 meters north. "
            "Water samples from the cluster tested within safe mineral ranges. The regional park "
            "authority accepted the reroute plan in a unanimous vote."
        ),
        "question": "Near which marker stone was the sinkhole cluster found?",
        "answers": ["22", "marker stone 22"],
    },
    {
        "id": "off_code_q1",
        "context": (
            "Release notes for firmware 4.2.1: fixed a race in the scheduler when two USB devices "
            "enumerated simultaneously. Known issue: CRC warnings may still appear on long cables "
            "over 3 meters. Workaround is to force USB 2.0 mode via config flag usb_force_hs=0. "
            "Support ID for this note is SUP-4419."
        ),
        "question": "What is the support ID for the firmware 4.2.1 release note?",
        "answers": ["SUP-4419"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small external eval slice")
    p.add_argument("--model", default=PRIMARY_MODEL_ID)
    p.add_argument("--source", default="auto", choices=["auto", "longbench", "offline"])
    p.add_argument("--n", type=int, default=60, help="Max items total")
    p.add_argument(
        "--tasks",
        default="multifieldqa_en,qasper,hotpotqa",
        help="LongBench jsonl stems (comma list)",
    )
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--window", type=int, default=SNAPKV_WINDOW)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--arms",
        default="full,valley,novelty,hybrid,query_hold",
        help="Comma list: full,valley,novelty,hybrid,query_hold",
    )
    p.add_argument(
        "--hold-budget",
        type=int,
        default=2048,
        help="Mid-stream hold cache for query_hold arm",
    )
    return p.parse_args()


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, answers: list[str]) -> float:
    return max((f1_score(pred, a) for a in answers), default=0.0)


def substring_hit(pred: str, answers: list[str]) -> bool:
    p = normalize_answer(pred)
    for a in answers:
        na = normalize_answer(a)
        if na and na in p:
            return True
    return False


def build_prompt(tokenizer, context: str, question: str, max_ctx: int) -> str:
    user = (
        f"Read the context and answer the question with a short factual answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    messages = [{"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user + "\n"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_ctx:
        # keep head of template + tail of context/question
        keep_head = max_ctx // 5
        keep_tail = max_ctx - keep_head
        ids = ids[:keep_head] + ids[-keep_tail:]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return text


_PAD_UNITS = [
    "Routine notes continue with weather observations and staffing reminders for the next shift. ",
    "Secondary paragraphs describe ordinary operations without introducing new proper nouns. ",
    "Additional background material expands length so streaming compression is forced. ",
    "The surrounding text is intentionally generic and should not contain answer keys. ",
]


def pad_context(tokenizer, context: str, target_tokens: int, seed: int) -> str:
    """Pad context with generic prose so total prompt exceeds stream budget."""
    import random

    rng = random.Random(seed)
    chunks = [context, "\n\n"]
    # leave room for question wrapper (~150 tokens)
    budget = max(target_tokens - 200, 256)
    while True:
        text = "".join(chunks)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n >= budget:
            break
        chunks.append(rng.choice(_PAD_UNITS))
    return "".join(chunks)


ROOT = Path(__file__).resolve().parents[1]
LONGBENCH_DIR = ROOT / "data" / "longbench" / "data"


def ensure_longbench_data() -> Path | None:
    """Download THUDM/LongBench data.zip if missing; return data/ dir with jsonl files."""
    if LONGBENCH_DIR.is_dir() and any(LONGBENCH_DIR.glob("*.jsonl")):
        return LONGBENCH_DIR
    try:
        from huggingface_hub import hf_hub_download
        import zipfile
    except Exception as e:
        print(f"hub import failed: {e}", flush=True)
        return None
    try:
        print("Downloading THUDM/LongBench data.zip ...", flush=True)
        zpath = hf_hub_download(
            repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset"
        )
        out = ROOT / "data" / "longbench"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        if LONGBENCH_DIR.is_dir() and any(LONGBENCH_DIR.glob("*.jsonl")):
            print(f"Extracted LongBench to {LONGBENCH_DIR}", flush=True)
            return LONGBENCH_DIR
    except Exception as e:
        print(f"LongBench download failed: {type(e).__name__}: {e}", flush=True)
    return None


def load_longbench_jsonl(
    tasks: list[str],
    n_per_task: int,
    max_ctx_chars: int = 50000,
) -> list[dict]:
    data_dir = ensure_longbench_data()
    if data_dir is None:
        return []
    items: list[dict] = []
    for task in tasks:
        path = data_dir / f"{task}.jsonl"
        if not path.is_file():
            print(f"  missing {path.name}", flush=True)
            continue
        n_task = 0
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_task >= n_per_task:
                    break
                row = json.loads(line)
                ctx = row.get("context")
                if not ctx:
                    continue
                ctx = str(ctx)
                q = str(row.get("input") or row.get("question") or "")
                ans = row.get("answers") or row.get("answer") or []
                if isinstance(ans, str):
                    ans = [ans]
                ans = [str(a) for a in ans if str(a).strip()]
                if not ans:
                    continue
                if len(ctx) > max_ctx_chars:
                    half = max_ctx_chars // 2
                    ctx = ctx[:half] + "\n...\n" + ctx[-half:]
                items.append(
                    {
                        "id": f"{task}_{row.get('_id', i)}",
                        "task": task,
                        "context": ctx,
                        "question": q or "Answer based on the context.",
                        "answers": ans,
                        "source": f"THUDM/LongBench:{task}",
                        "orig_length": row.get("length"),
                    }
                )
                n_task += 1
        print(f"  loaded {n_task} from {task}", flush=True)
    return items


def load_items(
    source: str,
    n: int,
    tokenizer=None,
    pad_to: int = 0,
    tasks: str = "multifieldqa_en,qasper,hotpotqa",
) -> tuple[list[dict], str]:
    items: list[dict] = []
    src = "offline"
    if source in ("auto", "longbench"):
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        # split n across tasks
        n_per = max(1, (n + len(task_list) - 1) // len(task_list))
        print(f"Loading LongBench local jsonl tasks={task_list} n_per={n_per}", flush=True)
        lb = load_longbench_jsonl(task_list, n_per_task=n_per)
        if lb:
            items, src = lb[:n], "longbench"
        elif source == "longbench":
            print("LongBench unavailable", flush=True)
            return [], "longbench"
        else:
            print("Falling back to offline external-style items", flush=True)
    if not items:
        items = [dict(x) for x in OFFLINE_ITEMS[:n]]
        src = "offline"

    # Pad only short offline contexts so stream@512 compresses
    if tokenizer is not None and pad_to > 0 and src == "offline":
        for i, it in enumerate(items):
            seq0 = len(tokenizer.encode(it["context"], add_special_tokens=False))
            if seq0 < pad_to - 200:
                it["context"] = pad_context(
                    tokenizer, it["context"], pad_to, seed=i + 7
                )
                it["padded"] = True
    return items, src


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=== External slice eval ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pad offline items to max_ctx so stream budgets matter
    items, src = load_items(
        args.source,
        args.n,
        tokenizer=tokenizer,
        pad_to=args.max_ctx,
        tasks=args.tasks,
    )
    if not items:
        print("No items to evaluate.", flush=True)
        return

    print(
        f"source={src} n={len(items)} budget={args.budget} arms={arms} "
        f"max_ctx={args.max_ctx} tasks={args.tasks}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    rows: list[dict] = []

    for item in items:
        prompt = build_prompt(
            tokenizer, item["context"], item["question"], args.max_ctx
        )
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        seq_len = int(input_ids.shape[-1])
        print(
            f"\n--- {item['id']} seq={seq_len} q={item['question'][:80]!r} ---",
            flush=True,
        )

        for arm in arms:
            try:
                t0 = time.perf_counter()
                if arm == "full":
                    past, logits = prefill_chunked(model, input_ids, chunk_size=512)
                    peak = cache_seq_len(past)
                elif arm == "valley":
                    past, logits, st = prefill_streaming_valley(
                        model,
                        input_ids,
                        stream_budget=args.budget,
                        final_budget=args.budget,
                        chunk_size=512,
                        window_size=args.window,
                    )
                    peak = st.get("peak_cache_tokens", st.get("peak_cache"))
                elif arm == "novelty":
                    past, logits, st = prefill_streaming_novelty_pin(
                        model,
                        tokenizer,
                        input_ids,
                        stream_budget=args.budget,
                        final_budget=args.budget,
                        chunk_size=512,
                        window_size=args.window,
                    )
                    peak = st.get("peak_cache")
                elif arm == "hybrid":
                    past, logits, st = prefill_streaming_hybrid_pin(
                        model,
                        tokenizer,
                        input_ids,
                        stream_budget=args.budget,
                        final_budget=args.budget,
                        chunk_size=512,
                        window_size=args.window,
                    )
                    peak = st.get("peak_cache")
                elif arm == "query_hold":
                    past, logits, st = prefill_streaming_query_hold(
                        model,
                        tokenizer,
                        input_ids,
                        final_budget=args.budget,
                        hold_budget=args.hold_budget,
                        chunk_size=512,
                        window_size=args.window,
                    )
                    peak = st.get("peak_cache")
                else:
                    raise ValueError(arm)

                toks = greedy_generate(
                    model,
                    past,
                    logits,
                    args.max_new,
                    eos_id=tokenizer.eos_token_id,
                    next_position=seq_len,
                )
                pred = tokenizer.decode(toks, skip_special_tokens=True)
                f1 = best_f1(pred, item["answers"])
                hit = substring_hit(pred, item["answers"])
                dt = time.perf_counter() - t0
                row = {
                    "id": item["id"],
                    "source": item.get("source", src),
                    "arm": arm if arm == "full" else f"{arm}@{args.budget}",
                    "seq_len": seq_len,
                    "f1": round(f1, 4),
                    "substr_hit": hit,
                    "peak_cache": peak,
                    "seconds": round(dt, 3),
                    "pred": pred[:200].replace("\n", " "),
                    "gold": " | ".join(item["answers"])[:200],
                }
                rows.append(row)
                print(
                    f"  {row['arm']}: f1={f1:.2f} hit={hit} peak={peak} t={dt:.1f}s "
                    f"pred={row['pred'][:80]!r}",
                    flush=True,
                )
                del past
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                rows.append(
                    {
                        "id": item["id"],
                        "arm": arm,
                        "f1": 0.0,
                        "substr_hit": False,
                        "status": f"ERR:{type(e).__name__}",
                        "pred": str(e)[:120],
                    }
                )
                print(f"  {arm}: ERR {e}", flush=True)

    # Summary
    print("\n=== SUMMARY (mean F1 / substr hit rate) ===", flush=True)
    summary = []
    for arm in arms:
        label = arm if arm == "full" else f"{arm}@{args.budget}"
        arm_rows = [r for r in rows if r.get("arm") == label or r.get("arm") == arm]
        if not arm_rows:
            continue
        mean_f1 = sum(float(r.get("f1") or 0) for r in arm_rows) / len(arm_rows)
        hit_rate = sum(1 for r in arm_rows if r.get("substr_hit")) / len(arm_rows)
        print(
            f"  {label}: mean_f1={mean_f1:.3f} substr_hit={hit_rate:.3f} n={len(arm_rows)}",
            flush=True,
        )
        summary.append(
            {
                "arm": label,
                "mean_f1": round(mean_f1, 4),
                "substr_hit_rate": round(hit_rate, 4),
                "n": len(arm_rows),
            }
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"external_slice_{ts}.json"
    out_csv = RESULTS_DIR / f"external_slice_{ts}.csv"
    payload = {
        "ts": ts,
        "model": args.model,
        "source": src,
        "budget": args.budget,
        "max_ctx": args.max_ctx,
        "summary": summary,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
