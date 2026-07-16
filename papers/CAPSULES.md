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
| stream_valley | 33% | 67% | 93% |
| capsules (scored discovery) | 33% | 33% | 87% |
| **stream_oracle_pin** (perfect discovery) | **100%** | **100%** | — |

**VERDICT: `DISCOVERY_IS_THE_GAP`** (`capsules_20260716T162035Z`)

- Perfect discovery (force critical±R whenever still in cache) → **15/15** at stream **@512** multi-seed.  
- Same peak budget where valley is **5/15**.  
- Oracle retention = 1.0 every cell.  
- Therefore: **stream peak budget is enough**; **finding** the neighborhood online is not.

### What we learned

1. Atomic packing / sticky / pin-on-exit **without true facts** cannot beat valley.  
2. **If discovery were solved, stream@512 would match H1 multi-seed** — huge systems implication.  
3. Mid-stream attention is a **bad query-unknown detector** on this suite.  
4. The fresh research target is now precise: **query-unknown fact discovery under peak-KV**, not another packer.

### Next

| Priority | Work |
|----------|------|
| **1** | New detector (not last-window attn alone): multi-probe, novelty, entity-ish, small auxiliary model |
| 2 | Keep oracle_pin as regression upper bound in CI |
| 3 | Package H1 multi-seed + this gap as paper core narrative |
## What would count as a win

- Higher multi-seed success than `seed_valley` at the **same** stream budget, or  
- Same success at **lower** peak cache, or  
- Clear ablation: atomic packing helps; random contiguous chunks don’t  

## What would count as a loss (honest) — **we hit this on v0**

- Capsules ≲ valley at all budgets → atomicity alone insufficient; need better **query-unknown discovery**  

## Relation to prior work

Not “new SnapKV.”  
**New compression unit + explicit query-unknown problem statement** grounded in our kill tests.
