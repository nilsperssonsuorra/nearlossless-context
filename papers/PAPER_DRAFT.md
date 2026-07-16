# Near-Lossless Long Context under Fixed VRAM via Critical-Span Retention

**Status:** draft skeleton for portfolio / possible workshop note  
**Lab:** RTX 3090 24 GB · primary `Qwen/Qwen3-4B-Instruct-2507`  
**Repo:** nearlossless-context (private)  
**Last updated:** 2026-07-16

> Numbers from `results/paper_rigor_20260716T120146Z` (5 seeds × 3 depths) unless noted.

---

## Abstract (draft)

Long-context inference is limited by KV cache memory. Training-free token eviction methods reduce cache size, but it is unclear *which* tokens are necessary for near–full-KV quality. On a multi-seed controlled retrieval suite (5×3 cells, Qwen3-4B), we show that **retaining critical fact tokens plus a small local radius** \(R^*=1\) (with sinks and a recent/question window) is necessary and sufficient for ε≈0 quality—**15/15** oracle success, **0/15** anti-oracle—while full KV is **15/15**. A non-oracle attention scorer (`seed_valley`) matches ε=0 at **192** tokens on all cells (~**1.24×** mean oracle size ~149; mean per-cell tax ~**1.16×**). Under online streaming, multi-seed hay exposes a **query-unknown discovery gap**: attention stream@512 is only **33%** at 4k while perfect online pin of critical±R is **15/15** at the same peak budget. A **surface novelty detector** (rarity + digit/ID-like cues, no question) closes most of that gap (**93%** multi-seed@4k; **100%** at 8k and 16k multi-seed 3×3) and keeps peak cache **~1k** tokens through 16k—where valley often still needs **~2k**. The critical-span mechanism **transfers** across Qwen3, Qwen2.5, Llama-3.2, and hybrid Gemma-4 (full layers). Limitations: needle-class suite; residual scorer tax; novelty aligned with structured secrets; modest multi-seed \(N\).

---

## 1. Introduction

- **Problem:** KV scales with \(L\); local 24 GB forces short comfortable contexts.
- **Goal:** maximize \(L_\varepsilon = \max\{L : Q(M,L) \ge (1-\varepsilon) Q(\mathrm{Full},L)\}\) under peak VRAM / peak cache constraints; ε=0 on retrieval-critical tasks.
- **Claim:** near-lossless training-free compression is closer to **guarantee critical local neighborhoods** than to uniform thinning or pure quantization.

## 2. Related work (sketch)

- KV eviction / SnapKV / H2O / pyramid / hybrid attention (Gemma-style sliding+full).
- Quantization (KIVI, etc.) — complementary; we use fake-int8 only for byte accounting.
- **Positioning:** mechanism study + systems \(L_\varepsilon\) on a fixed workstation, not a new SOTA leaderboard chase.

## 3. Hypothesis H1 / H1′

**Necessity:** Dropping tokens that encode the fact → retrieval fails (ε=0).  
**Sufficiency (H1′):** sinks ∪ critical±\(R^*\) ∪ recent/question window restores full-KV success; bare critical tokens without radius often corrupt the fact (e.g. trailing digit).

**Kill protocol:** full / oracle_r1 / anti_oracle / recent.

| Result (5×3 multi-seed @4k) | Rate |
|-----------------------------|------|
| full | **15/15 (100%)** |
| oracle_r1 | **15/15 (100%)** |
| anti_oracle | **0/15 (0%)** |

\(R^*=1\); mean |oracle| ≈ 149 tokens (~155 start/mid, ~136 end) vs full ~4k.

## 4. Scorer tax

Without oracle labels, `seed_valley` (shared attention scores → seeds → contiguous valley + ±R) targets the oracle set.

| Method | \(B_{\min}\) all cells | Mean per-cell \(B_{\min}\) | Tax vs mean \|oracle\| |
|--------|------------------------|----------------------------|-------------------------|
| oracle_r1 | ~149 (exact set) | — | **1.0×** |
| seed_valley | **192** | **172** | **~1.24×** global / **~1.16×** mean |
| @176 Jaccard vs oracle | — | 0.83 | ~29 extra non-oracle tokens |

Depth 0 hardest (often 192); end-depth often 155.

## 5. Systems: streaming \(L_\varepsilon\)

Posthoc compress still peaks at full prefill KV. **Online** compress after chunks caps peak cache ≈ stream_budget + chunk.

**Multi-seed @4k:** scored stream@512 is **not** robust (33%); stream@1536 ~93%.

**Critical upper bound:** with *perfect discovery* (oracle pin of critical±R online), stream@**512** is **15/15** multi-seed — same peak budget where scored attention methods fail. So the systems limit is **not** peak cache; it is **query-unknown discovery**.

**Partial close of the gap:** a **surface novelty detector** (rarity + digit/ID-like tokens, no question) reaches **14/15 (93%)** at stream@512 multi-seed — matching the quality of valley@1536 at ~⅓ the stream budget on this suite.

| Setting | Multi-seed @4k (5×3) |
|---------|----------------------|
| stream_valley@512 | 33% |
| stream_oracle_pin@512 | **100%** |
| stream_novelty@512 | **93%** |
| stream_valley@1536 | ~93% |

**Long \(L\) multi-seed (3×3 cells):** novelty@512 remains strong where valley@512 stays end-only:

| \(L\) | valley@512 | novelty@512 | valley@1536 |
|-------|------------|-------------|-------------|
| 8k | 33% | **100%** | 89% |
| 12k | 33% | **78%** | 78% |
| 16k | 33% | **100%** | 78% |

Peak cache **~1k** (stream@512) vs **~2k** (stream@1536). Stress suite (NL names/places, ID-flood filler, multi-needle) also favors novelty over valley at fixed@512.

## 6. Transfer

| Model | H1 | Stream (4k mid) | Posthoc min |
|-------|----|-----------------|-------------|
| Qwen3-4B | holds | 512 | ~176 |
| Qwen2.5-3B | holds | 512 (768@8k) | ~320 |
| Llama-3.2-3B | holds | 512 | ~256 |
| Gemma-4 E4B hybrid | holds (full layers) | novelty@512 **9/9** multi-seed; valley@1024 | ~176 |

Hybrid lesson: compress full layers only; score-pass must keep sliding `cumulative_length` as absolute prefix (HF `q_offset` from layer 0). Novelty discovery transfers to hybrid at primary-class stream budgets.

## 7. Adaptive policy

`prefill_auto`: L-based schedule + optional entity estimate + **per-model floors** (Gemma stream≥1024 R≥2, etc.).

## 8. Limitations

1. Needle / multi-needle suite — not general long-context benchmarks (RULER, LongBench, …).  
2. Residual scorer tax; stream ≠ oracle-tight.  
3. Fake int8 logical bytes, not production kernels.  
4. Multi-seed N still modest for formal claims.  
5. Generation uses greedy decode; template sensitivity possible.

## 9. Conclusion

Critical-span retention with small local radius is a **necessary structure** for near-lossless training-free KV compression on retrieval tasks; non-oracle scorers approximate it with small tax; streaming raises practical \(L_\varepsilon\) under 24 GB; the mechanism transfers across families including hybrid models.

---

## Reproducibility

```text
python experiments/bench_paper_rigor.py --seeds 0,1,2,3,4 --ctx 4096
python experiments/bench_transfer.py --model <id>
python experiments/bench_adaptive_e2e.py
```

Primary: `Qwen/Qwen3-4B-Instruct-2507`, transformers + CUDA bf16, RTX 3090.
