# Usage — near-lossless long context on a 3090

Primary model: `Qwen/Qwen3-4B-Instruct-2507`  
Goal: larger usable \(L\) under ~24 GB with small decode KV.

## One-call API

```python
from compress_adaptive import prefill_auto
from decode_utils import greedy_generate

# Long single-document retrieval (default streaming)
past, logits, info = prefill_auto(model, input_ids, mode="stream")
# info["policy"] has R, stream_budget

# Multi-secret / multi-doc: pass entity count (or safe_multi)
past, logits, info = prefill_auto(
    model, input_ids, mode="stream", n_entities=3
)
# equivalent:
# prefill_auto(model, input_ids, mode="stream", safe_multi=True)
# optional logical int8 accounting (still dequants for HF):
# prefill_auto(model, input_ids, mode="posthoc", use_int8=True)

# Best scorer quality (higher peak VRAM during prefill)
past, logits, info = prefill_auto(model, input_ids, mode="posthoc")

# Full KV gold (chunked prefill)
past, logits, info = prefill_auto(model, input_ids, mode="full")

toks = greedy_generate(
    model, past, logits, max_new,
    eos_id=tokenizer.eos_token_id,
    next_position=input_ids.shape[-1],  # critical after compress
)
```

## Practical length guide (single-needle, measured)

| Target L | Recommended | Notes |
|----------|-------------|--------|
| ≤ 4k | `stream` or `posthoc` | posthoc@176 tiny decode KV |
| ≤ 8k | `stream` (auto → 512) | comfortable |
| ≤ 24k | `stream` (auto → **1536**) | **reliable** peak ~2k / ~9 GB |
| ≤ 40k | `stream` (auto → **2048** if L≥28k) | reliable long single-needle |
| Multi-secret | `stream` (auto-raise or `safe_multi=True`) | mid-stream peak probe can raise budget |

## Do not

- Use **stream@512** for multi-hop with distractors (can return wrong secrets).  
- Forget `next_position=original_prompt_len` after compression (RoPE).  
- Expect 24k multi-needle quality with single-needle budgets.

## Benches

```powershell
python experiments\bench_adaptive_e2e.py --ctx 4096
python experiments\bench_h3_hop3.py --ctx 4096
python experiments\bench_l_epsilon.py --lengths 4096,8192 --allow-long --mid-only
```

See `results/FINDINGS.md` for full tables and verdicts.
