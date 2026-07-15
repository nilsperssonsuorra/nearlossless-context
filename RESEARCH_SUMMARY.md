# Research summary — nearlossless-context

**Lab:** RTX 3090 24 GB · `Qwen/Qwen3-4B-Instruct-2507`  
**Question:** How do we raise usable context length \(L\) at near–full-KV quality under fixed VRAM?

## Mechanism

| Hypothesis | Result |
|------------|--------|
| **H1** Critical spans + local radius \(R^*\) | Supported; bare spans fail; \(R^*=1\) single-needle; larger \(R\) helps multi-entity scorers |
| **H2** Priority beats volume at fixed bytes | Supported |
| **H3** Multi-needle / multi-hop | Mechanism holds; scorers need higher budget/R; stream@512 unsafe with distractors |

**Law (working):** near-lossless retrieval under training-free KV compression ≈ retain **critical local neighborhoods** (and all of them if multi-entity), not uniform thinning.

## Systems results

| Setting | Outcome |
|---------|---------|
| Posthoc seed_valley @176 | ~40× smaller decode KV @8k mid |
| Stream @512 | \(L_\varepsilon=8\mathrm{k}\) all depths |
| Stream @1536 | Reliable **~24k**; observed **≥40k** single-needle; peak cache ~2k, VRAM ~9 GB |
| Stream ≥28k mid | Prefer **@2048** (mid can flake at 1536) |
| Multi-needle | Posthoc R=8@384; stream R=8@1024 |
| 3-hop + distractors | Supported with multi schedule; stream L-only can return wrong secret |
| Adaptive posthoc | Auto n̂ (sink-masked peaks) + schedule: single/multi/hop3 pass |
| Adaptive stream | Needs n prior or mid-stream auto-raise; L-only not enough for multi |

## Practical “how much fits?”

| Before (this machine) | After |
|----------------------|--------|
| ~4k comfortable full KV; 8k laggy | ~**24k** reliable near-lossless needle (stream); **≥40k** observed |
| Decode KV grows with L | Decode stays ~**0.2 GB** class at stream budgets |

**~6–10× longer** single-needle context vs old comfortable 4k, under similar peak VRAM (~9 GB).

## Code entrypoints

```python
from compress_adaptive import prefill_auto

past, logits, info = prefill_auto(model, input_ids, mode="stream")
past, logits, info = prefill_auto(model, input_ids, mode="stream", safe_multi=True)
past, logits, info = prefill_auto(model, input_ids, mode="posthoc")
```

See `USAGE.md` and `results/FINDINGS.md`.

## What this is / isn’t

**Is:** measured mechanism + systems on primary 4B; **H1 transfers** to Qwen2.5-3B, Llama-3.2-3B, and hybrid Gemma-4 E4B (full layers).  
**Isn’t:** general long-context SOTA, true int8 kernels, oracle-tight stream budgets on every model without retune.

## Transfer

| Model | H1 | stream | posthoc min (4k mid) |
|-------|----|--------|----------------------|
| Qwen3-4B (primary) | holds | @512 4k ok; long L uses 512–2048 | ~176 |
| Qwen2.5-3B | holds | @512 4k ok; 8k needs **768** | ~**320** |
| Llama-3.2-3B | holds | **@512 4k+8k ok** | ~**256** |
| Gemma-4 E4B (hybrid) | holds (full layers) | **@1024 R=2** (not 512) | ~**176** (after hybrid score-pass fix) |

## Adaptive policy (per-model)

`prefill_auto` applies transfer floors from `model.config` (override with `model_id=`):

- **Gemma-4** stream → R≥2, budget ≥1024  
- **Qwen2.5** posthoc ≥320; stream ≥768 @ L≥8k  
- **Llama-3.2** posthoc ≥256  

## Next

1. Residual scorer tax → oracle (primary)  
2. Public note / blog with tables above  
3. Optional: true int8 kernels / multi-seed transfer variance  
