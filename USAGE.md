# Usage — near-lossless long context on a 3090

Primary model: `Qwen/Qwen3-4B-Instruct-2507`  
Goal: larger usable \(L\) under ~24 GB with small decode KV.

## Runnable example

After `pip install nearlossless-context`, run:

```powershell
nearlossless-example
```

The supported Transformers range is `>=4.57.1,<5`; Transformers 5 compatibility
has not yet been validated against the mutable KV-cache internals.

The default `Qwen/Qwen2.5-0.5B-Instruct` model is intended as an accessible API
demonstration, not as a reproduction of the paper's quality results. The command
uses CUDA automatically when available; on CPU it uses float32 and a shorter
prompt, so expect slower inference and roughly 2 GB or more of available RAM.
Use the paper's primary Qwen3-4B model and the documented experiment commands for
research reproduction.

Useful overrides:

```powershell
nearlossless-example --device cpu --prompt-tokens 768
nearlossless-example --device cuda --prompt-tokens 4096 --max-new-tokens 32
nearlossless-example --help
```

## One-call API

```python
from nearlossless_context import greedy_generate, prefill_auto

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

## Public metadata reference

`prefill_auto` keeps its backward-compatible return shape:
`(past_key_values, last_logits, info)`. The exported `PrefillInfo`, `PolicyInfo`,
and `CompressionStats` `TypedDict` definitions describe the dictionaries for
type checkers and IDEs.

Every `info` dictionary has these stable fields:

| Field | Type | Meaning |
|---|---|---|
| `path` | `str` | Concrete prefill/compression path used |
| `L` | `int` | Original prompt length in tokens |
| `policy` | `PolicyInfo \| None` | Resolved policy; `None` only for `full` |
| `model_id` | `str \| None` | Detected or explicitly supplied model ID |

Mode-specific stable fields:

| Mode | Additional fields |
|---|---|
| `stream` | `discovery`, `n_entities_hat`, `stats`, `logical_kv_mb_int8`, `use_int8` |
| `posthoc` | `n_entities_hat`, `keep_count`, `cache_tokens`, `logical_kv_mb_int8`, `use_int8` |
| `full` | `cache_tokens` |

All stream `stats` dictionaries contain `stream_budget`, `final_budget`, and
`n_compress`. Discovery-specific measurements such as `peak_cache`,
`final_cache`, or `peak_cache_tokens` are optional fields in
`CompressionStats`. `PolicyInfo` contains `R`, `budget`, `stream_budget`,
`mode`, `note`, and `family`.

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
