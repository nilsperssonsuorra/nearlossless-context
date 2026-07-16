# Fact capsules + query-unknown streaming

**Status:** active research direction (fresh problem framing)  
**Builds on:** H1′ (critical ± local radius), multi-seed stream failures  

---

## The new object

A **fact capsule** is an atomic contiguous span `[lo, hi]` in cache index space:

- discovered from attention peaks (query-unknown) or oracle critical±R  
- **keep all tokens or drop the capsule entirely**  
- never partially fill a neighborhood to free slots for lower peaks  

This is the design consequence of H1′: the unit of near-lossless memory is a **neighborhood**, not a ranked token.

## The hard regime

| Regime | Question known? | Difficulty |
|--------|-----------------|------------|
| Posthoc compress | Yes (obs window = question) | Easier — scores point at facts |
| **Stream compress** | **No** (obs = local recent only) | **Hard** — multi-seed @512 failed (33%) |

**Query-unknown near-lossless memory:** retain potential facts under peak-KV budget before the user question exists.

## Method (v0)

1. Chunked prefill  
2. When cache > budget: score last window vs prefix (same as stream valley)  
3. Discover capsules: local maxima → valley grow → force ±R\* → merge overlaps  
4. Pack by score under budget with sinks + recent **forced**  
5. Atomic only — spare slots left empty rather than splitting a capsule  

Code: `experiments/capsules.py`  
Bench: `experiments/bench_capsules.py`

## Multi-seed results (primary @4k, 5×3)

| Arm | @512 | @1024 | @1536 |
|-----|------|-------|-------|
| stream_valley (attn) | 33% | 67% | 93% |
| capsules (attn discovery) | 33% | 33% | 87% |
| **stream_oracle_pin** | **100%** | **100%** | — |
| **stream_novelty** (surface detector) | **93% (14/15)** | **93%** | — |

**VERDICT: `DISCOVERY_IS_THE_GAP` → `NOVELTY_DETECTOR_OK` (v0)**  
(`capsules_20260716T162035Z`, `capsules_20260716T164126Z`)

- Perfect discovery → 15/15 @512.  
- **Surface novelty detector** (rarity + digits/ID-like tokens, no question) → **14/15 @512**, same peak class as valley that only gets 5/15.  
- Closes most of the discovery gap **without** final-question attention.

### What we learned

1. Peak@512 is enough for multi-seed when neighborhoods are retained.  
2. Mid-stream **attention** is a bad query-unknown detector; **surface novelty** works on this ID/code needle suite.  
3. Atomic capsules were a red herring until discovery worked.  
4. Stress suite: NL names/places **87%**, ID-flood filler **100%**, multi3 **80%** vs valley much lower (`novelty_stress_*`).  
5. Long-\(L\) multi-seed: sticky novelty@512 → **9/9 @16k / 24k / 32k / 40k** (non-sticky 24k was 6/9); valley@512 stays **33%** end-only.

### Code

- `experiments/novelty_detect.py` — `prefill_streaming_novelty_pin`  
- Default: `prefill_auto(..., mode="stream", discovery="novelty")`  
- Benches: `bench_capsules.py --novelty`, `bench_novelty_stress.py`, `bench_novelty_longL.py`

### Next

| Priority | Work |
|----------|------|
| 1 | ~~Sticky multi-seed ≥24k–40k~~ **done** — 9/9 @512 through **40k** |
| 2 | ~~Paper narrative + fig + limitations~~ **done** — `PAPER_DRAFT.md`, `figures/fig1_story.png` |
| 3 | Multi-entity packing / multi-hop under sticky novelty; optional external bench slice |
## What would count as a win

- Higher multi-seed success than `seed_valley` at the **same** stream budget, or  
- Same success at **lower** peak cache, or  
- Clear ablation: atomic packing helps; random contiguous chunks don’t  

## What would count as a loss (honest) — **we hit this on v0**

- Capsules ≲ valley at all budgets → atomicity alone insufficient; need better **query-unknown discovery**  

## Relation to prior work

Not “new SnapKV.”  
**New compression unit + explicit query-unknown problem statement** grounded in our kill tests.
