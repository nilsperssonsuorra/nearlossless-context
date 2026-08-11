# Critical-Span Retention and Query-Unknown Discovery for Long-Context KV Compression

**Status:** workshop / arXiv-style draft  
**Lab:** RTX 3090 24 GB · primary `Qwen/Qwen3-4B-Instruct-2507`  
**Repo:** nearlossless-context  
**Last updated:** 2026-07-18  

> Primary multi-seed tables: `results/paper_rigor_*`, `novelty_*`, `FINDINGS.md`.  
> **Figure:** [`figures/fig1_story.png`](figures/fig1_story.png) (regenerate: `python experiments/plot_paper_figures.py`).  
> **Readable PDF:** [`main.pdf`](main.pdf). Build it with `powershell -File papers/build_pdf.ps1` (LaTeX source: `main.tex`).

---

## Abstract

Long-context inference is limited by KV-cache memory. Training-free eviction methods shrink the cache, but it is often unclear *which* tokens a task requires and whether online streaming fails from insufficient peak budget or weak **query-unknown discovery**. We study these questions on a controlled retrieval suite using Qwen3-4B on an RTX 3090, then test the mechanism on a fixed public LongBench slice.

1. **Structure (H1′).** Within the measured 15-case protocol, retaining critical fact tokens plus radius \(R^*=1\) is empirically **necessary and sufficient** for exact single-fact recall: oracle **15/15**, anti-oracle **0/15**, and full KV **15/15**.
2. **Discovery gap.** Online streaming at moderate peak fails under multi-seed hay not because the budget is too small, but because discovery is weak: attention stream@512 is **33%**, while perfect online pin of critical±R is **15/15** at the same peak.
3. **Closing the gap (suite).** A **surface novelty detector** reaches **14/15** @4k. With sticky pin packing, it reaches **9/9** at each measured length from 16k through **40k**, with measured peak cache **~1k tokens**.
4. **Public long-doc QA (decomposition).** On a fixed 60-item LongBench slice, sticky novelty reaches 0.64× full mean F1 at peak ~1k. Posthoc query-aware compression reaches 0.95× at final 512 and ~1.01× at 1024, but requires peak \(L\). Query-hold's best measured point reaches 0.92× at peak ~2.5k.

The critical-span mechanism transfers across Qwen2.5, Llama-3.2, and hybrid Gemma-4 (full layers). We do **not** claim general long-context SOTA; see §Limitations.

---

## 1. Introduction

KV cache memory grows with context length \(L\). On a fixed workstation (here: 24 GB), full-cache decode becomes the bottleneck long before “the model cannot attend.” Training-free compressors (observation-window scoring, recent-only eviction, SnapKV-style selection) reduce tokens kept, but practice often optimizes average scores rather than **guaranteeing** the spans that encode answers.

We study **near-lossless** compression on retrieval-critical prompts: quality \(Q\) should match full KV at ε≈0 under a multi-seed protocol. The systems objective is

\[
L_\varepsilon \;=\; \max\{\, L : Q(M,L) \ge (1-\varepsilon)\,Q(\mathrm{Full},L) \,\}
\]

under **peak** cache / VRAM constraints, not only final decode size after a full prefill.

**Claim.** Near-lossless training-free KV compression is better understood as **retaining critical local neighborhoods** and **discovering them online before the question exists** than as uniform thinning or posthoc score ranking alone. On open long-document QA, the residual gap is largely an **online retention / peak-cache Pareto**, not a failure of query-aware scoring once the question is available.

---

## 2. Related work

- **Training-free eviction / selection:** H2O, StreamingLLM, Scissorhands / TOVA, SnapKV, pyramid keepers. Query-aware vs query-agnostic selection is active; our stream is query-unknown until the question arrives.
- **Streaming systems:** peak cache / VRAM under chunked prefill; hybrid sliding+full still leaves full-layer KV compressible. InfLLM-class page caches are complementary infrastructure.
- **Quantization:** KIVI-style KV quant is complementary; fake-int8 only for logical byte accounting.
- **Benchmarks:** needles / multi-hop synthetics vs LongBench slice (not full RULER/InfiniteBench leaderboards).

**Positioning.** Mechanism + discovery diagnosis + workstation \(L_\varepsilon\), not a new attention kernel or LongBench SOTA chase.

---

## 3. Methods

**Task.** Needle secrets in seeded filler; depths \(\{0,0.5,1\}\); exact-match greedy. Multi-seed \(5\times3\) @4k; \(3\times3\) long-\(L\).

**H1′ keep set.** \(K^\star =\) sinks ∪ critical±\(R\) ∪ recent/question. Kill arms: full, oracle_r1, anti_oracle. \(R^*=1\).

**Posthoc `seed_valley`.** Full prefill → obs-window (\(W{=}128\)) attention votes → seeds/valleys ±R → pack to \(B\); force sinks + last \(W\). LongBench posthoc UB: peak \(=L\).

**Streaming.** Chunk 512; compress when over budget; peak ≈ budget + chunk; absolute RoPE at decode.

**Oracle-online.** Force-keep critical±R at stream peak class.

**Surface novelty.** Prefix rarity + digit/ID cues → peaks ±R → **sticky** pin registry + score packing.

**Query-hold.** Hold larger mid-stream hybrid pins; query-aware tighten to final budget.

**Decode after compress.** Recompute first-token logits under compressed cache (re-forward last prompt token).

**LongBench slice.** 20× multifieldqa_en/qasper/hotpotqa @ max_ctx 4096; token F1 + substr hit; report ratios to same-item full.

**Suite risk.** Novelty favors surface-distinct secrets; LongBench is the honesty check.

**Default API.** `prefill_auto(..., discovery="novelty")` or `"query_hold"`.

---

## 4. Results

### Figure 1: Story in four panels

![Figure 1: H1 kill, discovery gap, long-L success, peak cache](figures/fig1_story.png)

**A. H1′ kill (4k, 5×3 multi-seed).** Full and oracle crit±1 both **15/15**; anti-oracle **0/15**. Bare critical tokens without radius often corrupt trailing digits (H1 needs local context); \(R^*=1\) restores ε≈0. Mean |oracle| ≈ **149** tokens vs full ~4k.

**B. Discovery gap (stream@512, same multi-seed protocol).** Attention valley **33%**; surface novelty **93%** (14/15); oracle pin **100%**. Peak budget is in the same class (~1k with chunk). Failures of valley at 512 are **detector** failures, not “stream cannot work.”

**C. Long \(L\) multi-seed (3×3).** Sticky novelty@512 is **9/9** at 16k, 24k, 32k, and **40k**. Valley@512 was measured at 8k, 12k, and 16k, where it reached 3/9; it was not run at longer lengths.

**D. Peak cache.** Full KV scales with \(L\); sticky novelty stream@512 stays **~1k** tokens at the three measured resource points. Prior valley operating points often used **~1.5 to 2k** peak for long \(L\).

### Scorer tax (posthoc, question known)

| Method | All-cell \(B_{\min}\) | Tax vs mean \|oracle\| |
|--------|----------------------|-------------------------|
| oracle_r1 | ~149 | 1.0× |
| seed_valley (multi-seed all-cell) | **192** | ~**1.29×** / mean cell tax ~**1.16×** |
| SnapKV (single-needle mid) | 192 | ~1.24× vs oracle mid |
| seed_valley (single-needle mid) | **176** | ~**1.14×** |

Posthoc is near-oracle-tight; **stream is not**, until discovery improves.

### Stress and transfer (summary)

| Check | Result |
|-------|--------|
| NL facts (names/places) sticky novelty@512 | **15/15** multi-seed |
| Code needles sticky | **15/15** |
| Adversarial ID-like filler (pre-sticky) | novelty **100%** vs valley **33%** |
| Multi-needle recall_all @512 | novelty **5/5** (`max_new≥96`); valley **0/5** |
| 2-hop (Alice→id→password) @512 | novelty **5/5**; valley **1/5** (needs @1024 for 5/5) |
| 3-hop + distractors @512 | novelty **5/5**; valley **0/5** (needs @1024 for 5/5) |
| Prose soft fact (no digits) @512 | novelty **15/15**; valley **53%** |
| Multidoc (6 titled docs) @512 | novelty **15/15**; valley/oracle_pin **5/15** |
| External-style mixed QA (10 items, ~4k padded) | novelty **10/10** hits (= full); valley **0/10** |
| **Public LongBench** (60: multifieldqa_en+qasper+hotpotqa @4k) | full F1 **0.28** / hit 23 to 25%; novelty **0.18 to 0.19** / 10%; valley **0.18** / 8%; hybrid **0.19** / **12%** |
| **Posthoc query-aware UB** (same 60; full prefill→seed_valley) | posthoc@512 **0.95×** full F1; @1024/2048 **~1.0×** (peak=\(L\)), showing that discovery *can* match full |
| **query_hold Pareto** (hold×final) | best **h2048→f1024 ≈ 0.92×** full F1 @ peak ~2.5k; novelty@512 **0.64×** @ peak 1k |
| Gemma-4 E4B hybrid novelty@512 | multi-seed **9/9** @4k (valley needs ~1024) |
| Qwen2.5 / Llama-3.2 | H1 holds; family-specific posthoc floors |

### LongBench decomposition (paper priorities 1 and 2)

Same 60 items, max_ctx=4096, Qwen3-4B greedy (`external_slice_20260717T222040Z`, `…T225301Z`).

**Posthoc upper bound** (question known; peak = full \(L\)):

| Arm | Mean F1 | Hit | F1 / full |
|-----|---------|-----|-----------|
| full | 0.282 | 25% | 1.00× |
| posthoc@512 | 0.267 | 27% | **0.95×** |
| posthoc@1024 | 0.284 | 28% | **~1.01×** |
| posthoc@2048 | 0.285 | 27% | **~1.01×** |

**Stream Pareto** (peak cache vs quality):

| Arm | Mean F1 | Hit | Mean peak | F1 / full |
|-----|---------|-----|-----------|-----------|
| novelty@512 | 0.179 | 10% | 1024 | 0.64× |
| query_hold h1024→f512 | 0.199 | 18% | 1536 | 0.70× |
| query_hold h1024→f1024 | 0.231 | 22% | 1536 | 0.82× |
| query_hold h2048→f512 | 0.212 | 17% | ~2531 | 0.75× |
| **query_hold h2048→f1024** | **0.260** | **22%** | **~2531** | **0.92×** |
| query_hold h4096→f1024 | 0.249 | 28% | ~3876 | 0.88× |

**Interpretation.** Open long-doc QA is not a failure of query-aware scoring (posthoc ≈ full). It is an **online retention** problem. query_hold makes the tradeoff quantitative: ~0.9× full F1 needs ~2.5k peak, not ~1k.

### Systems resources (peak VRAM / latency)

Mid-depth needle, seed 0 (`systems_resources_20260717T130143Z`):

| L | full peak VRAM | novelty@512 peak VRAM | full decode KV | novelty KV | novelty prefill |
|---|----------------|----------------------|----------------|------------|-----------------|
| 4k | 8.5 GB | **8.3 GB** | 569 MB | **72 MB** | 1.6 s |
| 16k | 10.6 GB | **8.3 GB** | 2296 MB | **72 MB** | 5.9 s |
| 40k | 14.5 GB | **8.3 GB** | 5754 MB | **72 MB** | 17.6 s |

Peak cache tokens stay **~1024** under novelty stream@512 at all three lengths. Full 40k prefill was extremely slow on this workstation session (~20 min; WDDM path); novelty remains interactive.

---

## 5. Adaptive policy (systems)

`prefill_auto` applies L-based budgets + per-model floors. With default `discovery="novelty"`, single-needle stream@**512** is the measured multi-seed operating point through **40k**. Attn-only discovery still needs larger floors (~1536 multi-seed @4k). Hybrid Gemma: compress **full** layers only; sliding layers keep absolute prefix bookkeeping for score-pass.

---

## 6. Limitations and what we do **not** claim

### 6.1 What we claim

- In the **15 measured cells** of the controlled 4k protocol, critical±\(R^*\) is empirically necessary and sufficient for exact single-fact recall.
- Stream failures at moderate budget are primarily a **query-unknown discovery** problem (oracle-online upper bound).  
- Sticky surface novelty **closes most of that gap** through **40k** multi-seed at peak cache ~1k on the primary model, with transfer smoke to other small instruct models including hybrid Gemma-4.  
- On a fixed LongBench slice, **posthoc reaches 0.95× full F1** at final B=512 (peak=\(L\)); online is a **Pareto** (query_hold best ≈ **0.92×** @ ~2.5k). Absolute full F1 is modest under 4B/greedy/4k truncate.

### 6.2 What we do **not** claim

| Not claimed | Why |
|-------------|-----|
| **General long-context SOTA** | Public LongBench slice (60 items) shows novelty well below full online (~0.18 vs 0.28 F1). Strong on needle/retrieval suites; not free lunch on open long-doc QA. |
| **All task types** | Primary suite is retrieval / needle-class (+ multi-needle, hops, prose, multidoc stress). Full reasoning chains, code repos, and full LongBench/RULER leaderboards untested. |
| **Oracle-tight online budgets for free** | Sticky multi3/hop is 5/5@512 with adequate decode; multi-secret packing still stresses oracle_pin@512 (2/5). Suite-alignment remains. |
| **Production memory stack** | Fake-int8 is logical accounting only; no fused CUDA kernels or serving productization. |
| **Novelty as universal “importance”** | Detector exploits surface distinctness vs repetitive filler. |
| **Large models / long training-free SOTA** | Primary evidence is ~3 to 4B instruct models on one GPU class. |
| **Statistical finality** | Multi-seed \(N\) modest (5×3 @4k; 3×3 long-\(L\)). Lab-grade. |
| **ε=0 beyond measured envelope** | Sticky multi-seed through **40k** (9/9 cells); longer \(L\) is extrapolation. |
| **Peak VRAM = final decode KV** | Streaming caps peak cache tokens; weights/activations dominate total VRAM. |

### 6.3 Threats to validity

- **Greedy decode** / chat templates vs exact-match and token F1; LongBench absolute full F1 ~0.28.  
- **Suite alignment:** surface novelty vs repetitive filler; LongBench is the honesty check.  
- **Multidoc packing:** novelty can beat labeled oracle_pin@512 (15/15 vs 5/15).  
- **Hybrid models:** only full layers compressed.  
- **Sample size** lab-grade; selection bias toward positive arms admitted.  
- **Community baselines:** SnapKV in posthoc scorer-tax; not full multi-seed stream LongBench arm.

---

## 7. Conclusion

The controlled experiments support a useful decomposition: retain a task-critical local neighborhood, then discover it before irreversible eviction. Sticky novelty reaches 9/9 at each measured length from 16k through 40k with a ~1k-token peak cache, but the LongBench slice reaches only 0.64× full mean F1 at that peak. The evidence supports a protocol-specific mechanism and an online peak-quality tradeoff, not general near-lossless compression.

---

## Reproducibility

```text
# Multi-seed H1 + scorer tax
python experiments/bench_paper_rigor.py --seeds 0,1,2,3,4 --ctx 4096

# Discovery gap / novelty
python experiments/bench_capsules.py --novelty --oracle-online --skip-capsules --budgets 512
python experiments/bench_novelty_stress.py
python experiments/bench_novelty_longL.py --ctx 16384,24576,32768,40960 --arms novelty --budgets 512

# Figure
python experiments/plot_paper_figures.py

# Transfer
python experiments/bench_transfer.py --model google/gemma-4-E4B-it
```

**Primary model:** `Qwen/Qwen3-4B-Instruct-2507`, transformers, CUDA bf16, RTX 3090.  
**Artifact index:** `results/FINDINGS.md`.
