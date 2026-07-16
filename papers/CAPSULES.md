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

## First multi-seed results (primary @4k, 5×3)

| Arm | @512 | @1024 | @1536 |
|-----|------|-------|-------|
| stream_valley | 33% | **67%** | **93%** |
| stream_capsules (atomic+fill+sticky) | 33% | 33% | 87% |

**VERDICT: `CAPSULES_NO_GAIN` (v0)** — atomic packing + sticky registry did **not** beat valley under query-unknown mid-stream scores.

### What we learned (this is still useful)

1. **Under-filling budget kills you** — pure atomic-only kept ~145 tokens and wasted slots (fixed with fill_remainder).  
2. **Bottleneck is discovery, not packing** — mid-stream obs windows rarely surface the true fact capsule; sticky pins the *wrong* peaks.  
3. **Depth=1 is easy** for both (fact in recent). Early/mid depths expose query-unknown failure.  

### Next experiments (if continuing this line)

| Idea | Why |
|------|-----|
| **Pin-on-exit** | When tokens *leave* the recent window, force a capsule if they ever scored high while local |
| **Coverage objective** | Maximize #capsules kept under budget, not only top score |
| **Oracle upper bound online** | If oracle capsules available mid-stream, how good is atomic pack? Isolates discovery vs packing |
| **Multi-probe scoring** | Score with several windows, not only the last chunk |

## What would count as a win

- Higher multi-seed success than `seed_valley` at the **same** stream budget, or  
- Same success at **lower** peak cache, or  
- Clear ablation: atomic packing helps; random contiguous chunks don’t  

## What would count as a loss (honest) — **we hit this on v0**

- Capsules ≲ valley at all budgets → atomicity alone insufficient; need better **query-unknown discovery**  

## Relation to prior work

Not “new SnapKV.”  
**New compression unit + explicit query-unknown problem statement** grounded in our kill tests.
