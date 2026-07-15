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

**Is:** measured mechanism + systems path on one modern 4B model; portfolio-grade research engineering.  
**Isn’t:** general long-context SOTA, multi-model transfer, true int8 kernels, or oracle-tight scorers on all tasks.

## Next

1. Harden stream-time n̂ (already: auto_raise_budget before first drop)  
2. Residual scorer tax → oracle keep size  
3. Second model family; optional true int8 attention  
4. Public note / blog with tables above  
