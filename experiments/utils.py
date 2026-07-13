"""Shared helpers for VRAM / timing on a single GPU."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch


def gpu_mem_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda(fn):
    """Run fn(), return (result, elapsed_seconds) with CUDA sync."""
    synchronize()
    t0 = time.perf_counter()
    out = fn()
    synchronize()
    return out, time.perf_counter() - t0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Union of keys so later rows can add fields (e.g. keep_count)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_prompt(tokenizer, target_tokens: int, filler: str, instruction: str) -> str:
    """Build a chat-ish prompt whose tokenized length is ~target_tokens."""
    # Rough fill then trim with the tokenizer
    est_chars = max(target_tokens * 4, 64)
    body = (filler * ((est_chars // max(len(filler), 1)) + 2))[:est_chars]
    messages = [
        {
            "role": "user",
            "content": (
                f"Context:\n{body}\n\n"
                f"{instruction}\n"
                "Answer in one short sentence."
            ),
        }
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = messages[0]["content"] + "\nAssistant:"

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > target_tokens:
        # Keep the end (instruction + generation prompt) — more realistic for long-ctx
        ids = ids[-target_tokens:]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    elif len(ids) < target_tokens:
        # Pad filler in the middle
        pad_n = target_tokens - len(ids)
        pad = tokenizer.encode(filler * (pad_n // 4 + 8), add_special_tokens=False)[
            :pad_n
        ]
        # Re-encode full text is messy; inject pad tokens into ids before last 64
        keep_tail = min(128, len(ids))
        ids = ids[:-keep_tail] + pad + ids[-keep_tail:]
        ids = ids[-target_tokens:]
        text = tokenizer.decode(ids, skip_special_tokens=False)
    return text
