# Cheap Long Context on a Single RTX 3090

**Status:** research brief (literature reviewed 2026-07-13)  
**Hardware:** RTX 3090 24 GB VRAM + 32 GB system RAM  
**Goal:** Invent a method that makes long context cheaper — higher usable context and less decode slowdown — without tanking quality.

---

## 1. One-sentence project claim (target)

> Under a **fixed byte budget** of KV memory (what actually limits a 3090), allocating **both token slots and bit-width per head/layer** beats uniform token budgets and uniform quant, yielding higher LongBench/RULER quality *and* flatter decode tok/s as context grows.

**Working name:** `ByteBudgetKV` (or `HeteroKV`)  
**Not:** “another SnapKV” or “Ada-KV but we rename it.”

---

## 2. Hardware reality (3090 + 32 GB)

| Resource | Budget | Implication |
|----------|--------|-------------|
| GPU VRAM | 24 GB | Weights + activations + KV must share this |
| System RAM | 32 GB | Limited CPU offload; don’t plan multi-100B full KV on CPU |
| Memory bandwidth | GDDR6X (not HBM) | Decode is **very** memory-bound; RocketKV notes consumer cards should benefit *more* than A100/H100 from KV traffic cuts |

### Model strategy: one model to prove the theory

**Rule:** Invent and validate on **exactly one primary model**.  
Multi-model is only a later “does it transfer?” check — not week 1–6.

#### Why small (2B/4B), not 27B/35B, for theory

| | Small (2B–4B) | Huge (27B+) |
|--|---------------|-------------|
| Iteration | Fast | Slow |
| VRAM for **long context** | Lots — can hit 32k–128k | Weights eat the card |
| Debug / log attention | Easy | Painful |
| Theory (budgets, bytes, quant) | Same math | Same math |
| Impressive demo | Later | After theory works |

KV compression is about **attention + memory**, not max model IQ.  
Prove ByteBudgetKV on one small model first.

#### Locked primary

| | Model |
|--|--------|
| **PRIMARY** | **Qwen3.5-4B** Instruct |
| Fallback only | **Qwen3.5-2B** if 4B flaky, or **9B** if 4B too weak to measure LongBench/NIAH gaps |

**Not in theory phase:** 27B, 35B-A3B, Gemma 4 mid/large, multi-model sweeps.

#### Later (only if phase 1 wins)

| Phase | Model | Why |
|-------|--------|-----|
| 2 | **One** of: Qwen3.6-27B Q4 *or* Gemma 4 26B-A4B | Real weight pressure on 24 GB |
| 3 | Optional second family | Transfer / appendix |

---

## 3. Literature notes (papers actually read)

### 3.1 SnapKV — *LLM Knows What You Are Looking For Before Generation*  
**arXiv:** [2404.14469](https://arxiv.org/abs/2404.14469) · NeurIPS 2024

| Item | Detail |
|------|--------|
| **Insight** | Important prompt KV positions are predictable from a short **observation window** at the end of the prompt; pattern is stable during generation |
| **Method** | Per-head vote from obs-window attention → top-k on prefix → **pooling/clustering** so neighbors of peaks are kept → concat with full obs window |
| **Claimed wins** | ~3.6× gen speed, ~8.2× memory efficiency @ 16k; ~380k tokens on A100-80GB HF with ~1k budget; LongBench near full-KV |
| **Limitation** | Permanent eviction; multi-query / multi-turn importance can shift; mostly **uniform token budget per head**; focuses on prompt compression more than decode-phase dynamics |
| **Code** | https://github.com/FasterDecoding/SnapKV |

**Takeaway for us:** Best single-stage permanent eviction baseline. Pooling matters (keeps local context integrity).

---

### 3.2 PyramidKV — *Dynamic KV Cache Compression based on Pyramidal Information Funneling*  
**arXiv:** [2406.02069](https://arxiv.org/abs/2406.02069)

| Item | Detail |
|------|--------|
| **Insight** | Attention **funnels** across layers: lower layers = broad/scattered attention; higher layers = sparse “massive activation” / sinks |
| **Method** | **Different KV budget per layer** (more lower, less higher), arithmetic pyramid; within layer, SnapKV-style score from instruction/local window + pooling |
| **Claimed wins** | ~12% KV ≈ full quality on LongBench; extreme budgets (e.g. 0.7%) large gains vs uniform methods; NIAH strong (even 128 slots on Llama-3-70B in paper setting) |
| **Limitation** | Layer shape is hand-designed (β hyperparam); still mostly **uniform across heads within a layer** in base PyramidKV |
| **Code** | https://github.com/Zefan-Cai/PyramidKV |

**Takeaway for us:** Layer-wise budgets are necessary; uniform depth is wrong.

---

### 3.3 Ada-KV — *Optimizing KV Cache Eviction by Adaptive Budget Allocation*  
**arXiv:** [2407.11550](https://arxiv.org/abs/2407.11550)

| Item | Detail |
|------|--------|
| **Insight** | Heads differ: sparse vs dispersed; **uniform head budgets waste memory** |
| **Theory** | L1 upper bound on pre/post-eviction attention output; Top-k minimizes bound for fixed per-head budgets; **global Top-B across heads** then count frequency → optimal head budgets under that bound |
| **Method** | Plug-and-play on SnapKV / Pyramid → Ada-SnapKV, Ada-Pyramid; safeguard α so sparse heads don’t get zero; GQA-compatible grouping; variable-length FlashAttention storage |
| **Evals** | RULER + LongBench; **question-aware vs question-agnostic** (critical — agnostic is harder and more realistic) |
| **Claimed wins** | Consistent quality gains especially low budget & question-agnostic |
| **Code** | https://github.com/FFY0/AdaKV |

**CRITICAL:** Ada-KV already owns **“adaptive token budget per head.”**  
Our earlier “HeadBudget” name **must not claim that as novel.** We **cite Ada-KV** and either:

1. use it as a strong baseline, or  
2. extend it (bytes, bits, decode dynamics, consumer metrics).

---

### 3.4 KIVI — *Tuning-Free Asymmetric 2-bit Quantization for KV Cache*  
**arXiv:** [2402.02750](https://arxiv.org/abs/2402.02750) · ICML 2024

| Item | Detail |
|------|--------|
| **Insight** | K has **channel-wise outliers** → quantize **K per-channel**; V is a sparse mixer → quantize **V per-token** |
| **Method** | 2-bit KIVI with residual full-precision window (streaming groups); fused dequant matmul |
| **Claimed wins** | ~2.6× peak mem; up to ~4× batch; 2.3–3.5× throughput; little quality loss vs FP16 on gen tasks |
| **Limitation** | Quantization only (no eviction); residual window needed for hard tasks (GSM8K); evals older models mostly |

**Takeaway:** Bits and axes of quantization matter as much as *which tokens*. Orthogonal to eviction → **combine**.

---

### 3.5 RocketKV — *Two-Stage KV Cache Compression*  
**arXiv:** [2502.14051](https://arxiv.org/abs/2502.14051) · ICML 2025

| Item | Detail |
|------|--------|
| **Insight** | Permanent eviction alone or dynamic sparse alone both miss oracle Top-k under low budgets; **unique** top-k tokens over a full decode are far fewer than sequence length |
| **Method** | Stage 1: coarse **SnapKV** permanent keep; Stage 2: **Hybrid Sparse Attention** (page min/max + head-dim sparsity à la Quest+SparQ); adaptive split of compression ratio across stages; **RocketKV-MT** keeps full storage for multi-turn but sparse decode |
| **Claimed wins** | Up to **400×** compression, **3.7×** e2e decode speedup on A100, ~32% peak mem save; strong LongBench/NIAH/RULER |
| **Models in paper** | Llama-3.1-8B-Ins, Mistral-7B-Ins-v0.2, LongChat-7B |
| **Code** | https://github.com/NVlabs/RocketKV |

**Takeaway:** Two-stage is SOTA-shaped for **decode speed**. We should **baseline RocketKV** if runnable. Note: expects even **more** relative gain on non-HBM consumer GPUs.

---

### 3.6 Related landscape (read/skim level)

| Work | Role |
|------|------|
| **StreamingLLM** | Sinks + recent window; infinite stream, not full memory |
| **H2O** | Heavy-hitters + recent during *generation*; weak on long *prompts* |
| **Quest / SparQ / Loki** | Dynamic sparse (keep storage, save bandwidth) |
| **DuoAttention / RazorAttention** | Retrieval heads vs streaming heads |
| **ChunkKV** | Evict/compress by **semantic chunks**, not isolated tokens (NVIDIA kvpress) |
| **KVzip** | Query-agnostic importance for multi-query reuse |
| **MiniKV** | Eviction + 2-bit hybrid |
| **MInference** | Prefill sparse patterns (TTFT) |
| **MLA (DeepSeek)** | Architectural KV compression (model design, not plug-in) |

**Tooling:** NVIDIA [kvpress](https://github.com/NVIDIA/kvpress) aggregates several compressors.

---

## 4. Gap analysis → what is still inventable

| Idea | Already done? | Still open? |
|------|---------------|-------------|
| Evict by obs-window attention | SnapKV | incremental only |
| Layer pyramid budgets | PyramidKV | refine schedule / learn schedule |
| **Head-wise token budgets** | **Ada-KV** | don’t re-claim |
| Two-stage permanent + dynamic sparse | RocketKV | hard to beat alone |
| Asymmetric K/V quant | KIVI | combine with eviction |
| Question-agnostic eval | Ada-KV, kvpress | we must include |
| Multi-turn permanent eviction pain | RocketKV-MT, SCBench | product-relevant |
| **Fixed byte budget (slots × bits)** | weakly explored | **YES — our wedge** |
| **Decode tok/s as primary metric on consumer GPU** | RocketKV on A100/H100; sparse on 3090 under-reported | **YES** |
| **Per-head / per-layer precision tiers** | rare | **YES** |
| Joint optimize **quality × tok/s / VRAM** Pareto on 3090 | not standard | **YES** |

### Refined invention (post-Ada-KV)

**ByteBudgetKV**

1. Total resource constraint is **bytes**, not tokens:  
   `sum_h (num_tokens_h × bytes_per_token_h) ≤ B_bytes`
2. Heads that are **sparse** get: few tokens, **higher precision** (q8/fp16 residual).  
   Heads that are **dispersed** get: more tokens, **lower precision** (q4/q2).
3. Layers get pyramid-style **byte** budgets (extend PyramidKV from slots → bytes).
4. Optional stage-2: RocketKV-style dynamic top-k inside the retained set for decode traffic.
5. Controller target for local product: **maximize quality s.t. peak VRAM ≤ 22 GB and decode tok/s ≥ τ** at context L.

**Why this can still be novel:**
- Ada-KV optimizes **token counts** under L1 attention-output bound.  
- KIVI optimizes **bit layouts** uniformly (same scheme all heads).  
- We optimize **heterogeneous (tokens, bits) under a byte budget** with **3090 bandwidth metrics**.

**Falsifiable claim:** At equal peak KV **bytes**, ByteBudgetKV > Ada-SnapKV (uniform bits) and > KIVI-only (no eviction) on RULER/LongBench **and** higher decode tok/s at 32k–128k.

---

## 5. Evaluation plan (hire-grade)

### Metrics (always report all four)

1. **Peak VRAM** (weights + KV + activations)  
2. **TTFT** (prefill) at L ∈ {4k, 16k, 32k, 64k, 128k}  
3. **Decode tok/s** at filled context L (this is the “doesn’t slow down” metric)  
4. **Quality:** LongBench avg + RULER + Needle-in-a-Haystack  

**Must include question-agnostic compression** (compress without seeing the question) — Ada-KV shows this is where methods die.

### Baselines (implement or wrap)

| Baseline | Notes |
|----------|--------|
| Full KV (fp16/bf16 or engine default) | Upper bound quality, lower bound length |
| KV quant only (q8 / q4 / KIVI-style if available) | llama.cpp `--cache-type-k/v` |
| StreamingLLM | Floor for “dumb” eviction |
| SnapKV | Permanent eviction SOTA class |
| PyramidKV | Layer budgets |
| Ada-SnapKV | Head token budgets — **mandatory** |
| RocketKV (if runnable) | Two-stage SOTA speed |

### Models (phased — do not parallelize)

| Phase | Model | Goal |
|-------|--------|------|
| **1 — invent** | **Qwen3.5-4B** only | Prove ByteBudgetKV theory + ablations |
| **2 — stress** | One of: Qwen3.6-27B Q4 **or** Gemma 4 26B-A4B | Real VRAM pressure (optional until phase 1 wins) |
| **3 — transfer** | Optional second family | Paper appendix only |

### Success criteria (v1 invention)

- [ ] At **same KV bytes**, quality ≥ Ada-SnapKV average on LongBench (±1 pt) **or** clearly better on RULER hard needles  
- [ ] At **same quality**, **≥1.5×** decode tok/s at 32k+ vs full KV (or vs Ada-SnapKV if full doesn’t fit)  
- [ ] On 3090: demonstrate **context length unlock** (e.g. 64k–128k usable where full KV OOMs or crawls)  
- [ ] Ablations: bytes vs tokens; hetero precision vs uniform q4; pyramid-bytes vs flat-bytes  
- [ ] Negative results documented  

---

## 6. 30-day execution plan

### Week 1 — Lab + tax curves (**Qwen3.5-4B only**)
- Env: CUDA, PyTorch, HF, FlashAttention if available  
- One model loaded; script: VRAM / TTFT / tok/s vs context  
- **Figure 1:** long-context tax on 3090 (small model → long L possible)

### Week 2 — Baselines on **same one model**
- Full KV, SnapKV, PyramidKV, Ada-style head budgets, KV quant  
- LongBench subset + NIAH (quality must be measurable on 4B — if not, bump to 9B only)  
- **Figure 2:** quality vs byte budget  

### Week 3 — Invention on **same one model**
- ByteBudgetKV: head → (n_tokens, n_bits) under fixed B_bytes  
- Ablations vs Ada-SnapKV at equal bytes  

### Week 4 — Harden  
- Question-agnostic; writeup  
- **Only if phase 1 wins:** optional one mid-size model run

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Invention collapses into Ada-KV + KIVI “and” | Need **joint byte objective** + ablations showing neither alone matches |
| Variable bits slow kernels on 3090 | Start fake-quant (accuracy) then real kernels; llama.cpp path for product demo |
| Qwen3.5/3.6 attention hooks differ | Budget engineering time for GQA / MoE gather-scatter |
| 32 GB RAM limits offload experiments | Focus GPU-resident compression first |
| Novelty bar rising fast (RocketKV, ChunkKV, …) | Own **consumer byte-budget Pareto** narrative + recent models |

---

## 8. Repo layout (this folder)

```
researchcontext/
  RESEARCH_BRIEF.md          ← this file
  papers/NOTES.md            ← short paper cards (expand over time)
  experiments/               ← scripts, configs (to create)
  results/                   ← csv + plots (to create)
```

---

## 9. Bottom line

- **Yes, invent something** — but not “HeadBudget” as pure token-per-head (Ada-KV, 2024).  
- **Yes, use recent families** — but **one small model first** (Qwen3.5-4B), not a zoo of 27B models.  
- **Yes, literature is read** — SnapKV, PyramidKV, Ada-KV, KIVI, RocketKV.  
- **Wedge:** **byte-heterogeneous KV** under fixed VRAM/bandwidth.

Next: scaffold bench on **Qwen3.5-4B only**.
