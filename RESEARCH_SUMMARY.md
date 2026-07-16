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
| Stream @512 | **Not multi-seed robust @4k (33%)** — depth=1 only |
| Stream @1536 | Multi-seed **~93% @4k**; long-\(L\) reliable **~24k** class; peak ~2k / ~9 GB |
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

## Multi-seed rigor (primary @4k, 5×3)

| Claim | Result |
|-------|--------|
| H1 full / oracle / anti | **15/15 / 15/15 / 0/15** |
| seed_valley all-cell \(B_{\min}\) | **192** (~1.24× oracle) |
| stream@1536 | **14/15** |

Draft: `papers/PAPER_DRAFT.md` · `experiments/bench_paper_rigor.py`

## Fresh line: fact capsules (query-unknown)

Atomic neighborhood objects + sticky absolute registry for stream compress.  
**First multi-seed result:** no gain vs seed_valley (discovery fails mid-stream). See `papers/CAPSULES.md`.

## Next

1. Capsule **pin-on-exit** / better query-unknown discovery — or kill the line cleanly  
2. Finish `papers/PAPER_DRAFT.md` + figure for H1 multi-seed (already solid)  
3. Multi-seed transfer slice  
