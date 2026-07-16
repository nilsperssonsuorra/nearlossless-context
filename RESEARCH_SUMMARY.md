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
| Stream valley @512 | **Not multi-seed robust** (33% @4k–16k) — end-depth only |
| Stream novelty @512 | **Multi-seed strong**: ~93% @4k (5×3); **100% @8k & 16k** (3×3); residual ~78% @12k; peak **~1k** |
| Stream valley @1536 | Multi-seed ~93% @4k; long-\(L\) ok-ish but **worse** than novelty@512 at 8k/16k multi-seed |
| Discovery upper bound | oracle_pin stream@512 → **15/15** @4k (**DISCOVERY_IS_THE_GAP**) |
| Stream ≥28k mid | Prefer **@2048** until re-measured with novelty (valley schedule) |
| Multi-needle | Posthoc R=8@384; novelty stream often wins valley @512 (stress suite 80%) |
| Adaptive | Default stream `discovery="novelty"`; single-needle stream@512 through 8k/16k |

## Practical “how much fits?”

| Before (this machine) | After |
|----------------------|--------|
| ~4k comfortable full KV; 8k laggy | Multi-seed **16k @ peak~1k cache** (novelty stream@512); valley long-L still **~24k–40k @~2k** |
| Decode KV grows with L | Decode stays small (stream budget); peak cache **flat in \(L\)** |

**~4× longer multi-seed** near-lossless needle at **half** the previous stream peak (1k vs 2k), vs old comfortable 4k full.

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
| Gemma-4 E4B (hybrid) | holds (full layers) | **novelty@512 multi-seed 9/9**; valley needs **@1024** | ~**176** (hybrid score-pass fix) |

## Adaptive policy (per-model)

`prefill_auto` applies transfer floors from `model.config` (override with `model_id=`):

- **Gemma-4** stream novelty ≥512 (valley/attn still ~1024)  
- **Qwen2.5** posthoc ≥320; stream ≥768 @ L≥8k  
- **Llama-3.2** posthoc ≥256  

## Multi-seed rigor (primary)

| Claim | Result |
|-------|--------|
| H1 full / oracle / anti @4k 5×3 | **15/15 / 15/15 / 0/15** |
| seed_valley all-cell \(B_{\min}\) | **192** (~1.24× oracle) |
| stream valley@512 / @1536 @4k | **33% / ~93%** |
| stream novelty@512 @4k | **93% (14/15)**; stress NL/adv/multi3 strong |
| stream novelty@512 long-L 3×3 | **9/9 @8k**, **7/9 @12k**, **9/9 @16k** |

Draft: `papers/PAPER_DRAFT.md` · benches: `bench_paper_rigor.py`, `bench_novelty_*.py`

## Fresh line: query-unknown discovery

| Method | stream@512 multi-seed |
|--------|----------------------|
| valley (attn) | 33% (code) / 0% multi3 |
| oracle_pin | 100% (code) |
| **novelty** | **100% code / 87% NL / 100% adv / 80% multi3** |

**Story:** H1 neighborhoods + small peak budget work; **discovery** was the gap; **surface novelty** closes it far beyond mid-stream attention (even under ID-like adversarial filler).

Default API: `prefill_auto(..., discovery="novelty")`.

## Next

1. Paper draft: H1 + discovery gap + novelty stress table  
2. Longer-L novelty multi-seed  
3. Optional: hybrid-model novelty transfer  
