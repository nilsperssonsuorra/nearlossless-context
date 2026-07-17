# Usage — near-lossless long context on a 3090

Primary model: `Qwen/Qwen3-4B-Instruct-2507`  
Goal: larger usable \(L\) under ~24 GB with small decode KV.

## One-call API

```python
from compress_adaptive import prefill_auto
from decode_utils import greedy_generate

# Long single-document retrieval (default: stream + novelty discovery)
past, logits, info = prefill_auto(
    model, input_ids, mode="stream", tokenizer=tokenizer
)
# discovery="novelty" (default) — query-unknown surface detector, stream@512-class
# discovery="hybrid" — novelty ∪ attn peaks + final query-aware re-rank (slight LB gain)
# discovery="query_hold" — hold ~2048 then query-aware tighten (best LB so far; higher peak)
# discovery="attn" — legacy mid-stream attention (needs larger budgets)
# info["policy"] has R, stream_budget, family; info["path"] shows novelty vs attn

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

# Optional model_id override for family floors (usually auto-detected):
# prefill_auto(model, input_ids, mode="stream", model_id="google/gemma-4-E4B-it")

toks = greedy_generate(
    model, past, logits, max_new,
    eos_id=tokenizer.eos_token_id,
    next_position=input_ids.shape[-1],  # critical after compress
)
```

**Family floors** (transfer-measured): Gemma-4 novelty stream ≥512 (valley/attn ~1024); Qwen2.5 posthoc ≥320 / stream ≥768 @8k+; Llama-3.2 posthoc ≥256.

## Practical length guide (single-needle, measured)

| Target L | Recommended | Notes |
|----------|-------------|--------|
| ≤ 4k | `stream` + **novelty** (default) | multi-seed ~**93–100%** @512; posthoc @192 |
| ≤ 40k | `stream` sticky novelty | multi-seed **9/9 @512** peak~**1k** (16–40k) |
| >40k | `stream` | auto ~**768** slack until measured |
| attn-only stream | `discovery="attn"` | needs **~1536** multi-seed @4k |
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
