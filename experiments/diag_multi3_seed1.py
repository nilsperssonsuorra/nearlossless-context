"""Diagnose multi3 seed=1: which keys survive novelty stream vs oracle."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_h1_oracle import find_minimal_span  # noqa: E402
from bench_novelty_stress import build_scenario_prompt, find_critical, score_keys  # noqa: E402
from capsules import prefill_streaming_oracle_pin  # noqa: E402
from config import PRIMARY_MODEL_ID  # noqa: E402
from decode_utils import greedy_generate  # noqa: E402
from novelty_detect import novelty_abs_set, prefill_streaming_novelty_pin  # noqa: E402
from snapkv import cache_seq_len  # noqa: E402


@torch.inference_mode()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    prompt, keys, _ = build_scenario_prompt(
        tok, scenario="multi3", target_tokens=4096, depth=0.5, seed=1
    )
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    ids = input_ids[0].tolist()
    seq_len = len(ids)
    crit = find_critical(tok, input_ids, keys)
    print("keys", keys)
    print("n_crit", len(crit), "seq", seq_len)

    for name, B in [("novelty", 512), ("novelty", 1024), ("oracle", 512)]:
        if name == "novelty":
            past, logits, st = prefill_streaming_novelty_pin(
                model,
                tok,
                input_ids,
                stream_budget=B,
                final_budget=B,
                max_capsules=24,
            )
        else:
            past, logits, st = prefill_streaming_oracle_pin(
                model,
                input_ids,
                critical=crit,
                stream_budget=B,
                final_budget=B,
                expand_radius=1,
            )
        # Map final abs if we can only see cache length
        S = cache_seq_len(past)
        print(f"\n=== {name}@{B} peak={st.get('peak_cache')} final={S} ===")
        # Offline pin coverage
        pin = novelty_abs_set(tok, ids, max_capsules=24, floor_ratio=0.65)
        for k in keys:
            span = find_minimal_span(tok, ids, k)
            offline = any(i in pin for i in span)
            print(f"  offline pin covers {k}: {offline} span={span[:5]}...")

        toks = greedy_generate(
            model,
            past,
            logits,
            80,
            eos_id=tok.eos_token_id,
            next_position=seq_len,
        )
        text = tok.decode(toks, skip_special_tokens=True)
        sc = score_keys(text, keys)
        print("  success", sc["success"], "hits", sc["hits"], sc["hit_map"])
        print("  ans:", sc["answer"][:200].replace("\n", " "))
        del past
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
