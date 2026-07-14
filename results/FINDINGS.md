# Findings (workstation-safe: max 4k context)

**Machine:** RTX 3090 24 GB (WDDM — avoid 8k+ full KV)  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`  
**Policy:** ctx ≤ 4096 for interactive full-KV jobs  

**Project goal:** maximize \(L\) at ε≈0 quality under 24 GB (see root README + `RESEARCH_BRIEF.md`).

---

## H1 kill experiment (`h1_oracle_20260713T214610Z`)

| Arm | Success | Meaning |
|-----|---------|---------|
| full | 100% | Gold |
| **oracle** (minimal fact tokens only) | **33%** | Often `…-199` typo — **bare span ≠ enough** |
| **oracle_ctx** (fact ±16 tokens) | **100%** | Local context restores full quality |
| anti_oracle (no fact tokens) | **0%** | Necessity holds |
| recent | 33% | Only when fact is at end |

**VERDICT: `H1_NEEDS_LOCAL_CONTEXT`**

Science takeaway: near-lossless single-fact retrieval under compression requires **critical tokens + a local neighborhood**, not global filler and not fact tokens alone. Oracle_ctx used ~150–200 cache tokens (~22–29 MB) vs full ~570 MB at 4k.

### H1′ radius sweep (`bench_h1_radius.py`)

Keep = sinks(8) ∪ critical±R ∪ recent(128). Full KV gold always ok.

| Depth | R_min (ε=0) | Notes |
|-------|-------------|--------|
| 0.0 | **1** | R=0 → `…-199` typo |
| 0.5 | **1** | same |
| 1.0 | **0** | fact already inside recent window |

**\(R^* = 1\)** covers all tested depths at 4k.  
At R=1: ~155 keep tokens, **~24 MB** KV vs full **~570 MB** (~24×).

---

## H2 equal-byte: priority vs volume (`h2_bytes_20260713T224808Z`)

**Question:** At fixed KV *bytes* (and fixed token count), does spending budget on **critical±R\*** beat keeping **more / equal irrelevant tokens**?

Arms (R\*=1, sinks=8, recent=128, L≈4k, 3 depths):

| Arm | Keep rule | Prec. | Success |
|-----|-----------|-------|---------|
| full | all tokens | bf16 | **100%** (3/3) |
| **priority_bf16** | sinks ∪ crit±1 ∪ recent | bf16 | **100%** (3/3) |
| **priority_int8** | same positions | int8 fake | **100%** (3/3) |
| volume_bf16_same_n | same *n*, uniform/mid, no crit priority | bf16 | 33% (1/3)† |
| volume_bf16_avoid_crit | same *n*, **forbid** critical | bf16 | **0%** (0/3) |
| volume_int8_2x_n | ~2× *n* (int8 ≈ same logical B) | int8 | 33% (1/3)† |
| volume_int8_2x_avoid | ~2× *n*, **forbid** critical | int8 | **0%** (0/3) |

† The sole volume success is depth=1.0, where the fact already sits in the **recent window** (recall=1.0 by chance of structure, not mid/early retrieval).

Budget scale: priority ~**136–155 tokens / ~19–22 MB** bf16 vs full **~570 MB**; priority_int8 logical ~**10–11 MB**.

**VERDICT: `H2_SUPPORTED`**

- Priority±R\* matches full at bf16 **and** int8.  
- Equal or double token volume that **avoids** critical spans always fails.  
- Extra tokens without critical local context do **not** substitute for the right positions (under this suite).

Science takeaway: under a byte budget, **which** tokens you keep (critical local context) dominates **how many** filler tokens you can stuff in (even at 2× via int8). Precision on the priority set can drop to int8 here without quality loss on the needle.

---

## Non-oracle scorer budget

**Question:** Without an oracle, can attention scoring recover critical±R\* at near-oracle budgets?

L≈4k, window=128, R\*=1, depths {0, 0.5, 1.0}. Min budget with **100% success on all depths**:

| Method | Min budget (ε=0) | vs oracle | Source |
|--------|------------------|-----------|--------|
| **oracle_r1** | **~155** | 1.00× | upper bound |
| **seed_valley** | **176** | **1.14×** | `scorer_budget_20260714T220943Z` |
| snapkv | 192 | 1.24× | same + earlier |
| seed_r1 (atomic ±R\*) | 208 | 1.34× | `…T220217Z` |
| snap_union_r1 | 208 | 1.34× | `…T220943Z` |
| shared_r\* (expand→recap) | 256 | 1.65× | `…T215259Z` |
| recent | none ≤512 | — | only depth=1 |

### What worked: valley completion (`seed_valley`)

1. Aggregate obs-window attention → shared prefix score (last layers, max over heads).  
2. Greedy high-score **seeds**.  
3. For each seed, keep the **contiguous high-score segment** (grow while score ≥ 0.25× seed) **plus** fixed ±R\*.  
4. Force sinks + recent; never break a segment to free slots for lower seeds.

Beats SnapKV by **16 tokens** (~29→27 MB). Scorer tax vs oracle: **21 tokens** (was 37 with SnapKV).

### What failed / weaker

- **expand→recap by token score** drops low-score neighbors of peaks (digits) → shared_r\* stuck at 256.  
- **Fixed ±R\* only** (`seed_r1`) helps mid-depth but not enough at depth=0.  
- **SnapKV per-layer union + ±R\*** does not beat vanilla SnapKV (union loses per-head diversity at decode? or over-complete wastes budget).

At tight budgets, **span recall predicts success**: recall≲0.8 → corrupted codes (`maple-qu-qu-19`, missing `7742`); recall≳0.87 → usually pass.

**VERDICT: `SCORER_NEAR_ORACLE`** (best non-oracle **seed_valley@176**)

- ~**21×** smaller KV than full (~27 MB vs ~570 MB).  
- Remaining tax **176→155** is pure **detection** (missing mass on some entity tokens), not “need more filler.”

---

## L_ε @ longer context (`l_epsilon_20260714T223432Z`, mid-depth)

**Setup:** chunked prefill (512), mid-depth needle only, lengths {4k, 6k, 8k}, Qwen3-4B, 3090.

| Method | L_ε (mid) | Decode KV @8k | Prefill peak VRAM @8k |
|--------|-----------|---------------|------------------------|
| **full** | **8192** | ~1148 MB | ~9.4 GB |
| **seed_valley@176** | **8192** | **~27 MB** | ~12.8 GB (score clone) |
| seed_valley@256 | 8192 | ~38 MB | ~12.8 GB |
| snapkv@256 | 8192 | ~38 MB | ~12.8 GB |
| snapkv@176 | 4096 | — | fails mid @6k/8k |
| recent@176/256 | none | — | mid needle fails |

**VERDICT: `L_EPS_MATCHED`** (mid-depth smoke through 8k)

- **compress_ε @8k** (decode KV): full / seed_valley@176 ≈ **1148 / 27 ≈ 42×** at ε=0 mid-needle.  
- Chunked prefill makes **8k full KV interactive** (~3s prefill, ~9.4 GB peak) — prior single-shot path thrashed WDDM.  
- seed_valley keeps **ε=0 quality** with oracle-near budget while SnapKV@176 breaks at longer L.  
- Caveat: prefill still builds full KV before compress (peak VRAM not reduced to 27 MB). Streaming / online eviction is the next systems step to raise *peak-VRAM-limited* L_ε further.

---

## Dense ceiling map (`ceiling_20260713T213158Z.csv`)

Full KV, mid-depth needle, no eviction:

| L (target) | actual | prefill | decode tok/s | peak VRAM | KV | needle |
|------------|--------|---------|--------------|-----------|-----|--------|
| 2048 | 1988 | 2.0 s | 14.2 | 9.3 GB | 280 MB | ok |
| 3072 | 3028 | 3.8 s | 13.6 | 10.9 GB | 426 MB | ok |
| 4096 | 4042 | 6.7 s | 11.5 | 13.2 GB | 568 MB | ok |

**Linear VRAM fit** (peak ≈ 5357 + 1.91 MB/token) → at **22 GB** usable:  
**\(L_{\max}\) est. ≈ 8.7k tokens** full bf16 KV (not yet measured; 8k may thrash WDDM).

Interpretation: weights + activations dominate base; KV grows ~1.9 MB/tok. Quality at 4k full KV is fine on single mid-needle. **Next for goal:** near-lossless KV quant / chunked prefill to approach ~8k+ without thrash, then hierarchy for beyond.

---

## Headline selection result (`equal_byte_20260713T212102Z.csv`)

Needle @ depths 0 / 0.5 / 1.0 × budgets 512 / 768 / 1024 / 1536:

| Method | Success | Mid-depth @512 |
|--------|---------|----------------|
| full | 100% (3/3) | ok |
| recent | 33% (4/12) | fail |
| **snapkv** | **100% (12/12)** | **ok** |
| **bytebudget** | **100% (12/12)** | **ok** |

### Equal-**slot** memory (budget=512, mid-depth, both correct)

| Method | tokens | runtime KV | logical KV |
|--------|--------|------------|------------|
| full | ~4057 | 571 MB | 571 MB |
| snapkv | 528 | **74 MB** | **74 MB** |
| **bytebudget** | 527 | 74 MB (dequant for HF) | **37 MB** (int8 account) |

### Equal-**logical-byte** read

- SnapKV @512 → **74 MB** logical, quality ok  
- ByteBudget @1024 → **73 MB** logical, **~2× tokens** (1039 vs 528), quality ok  

So under the same *logical* byte budget, ByteBudget can keep roughly **twice** the tokens (int8 vs bf16 accounting) while matching needle quality on this suite.

*(Runtime HF path still dequants to bf16 — true int8 kernels would also cut bandwidth.)*

---

## What fixed ByteBudget @512

Previous Ada path padded per-head lists to `max_k`, so effective length was ~200 tokens (starved mid needles).

**Now:** shared token set of size `keep_prefix` (same structure as SnapKV), built by:
- Ada-style per-head tops + max-vote union  
- **min_per_head floor**  
- sinks + cluster expand  
- then int8 logical quant + dequant for decode  

---

## Critical infrastructure fixes (still true)

1. DynamicCache `layers[i].keys/values`  
2. RoPE `position_ids` = true prompt length after compress  
3. Full prefill + score-on-clone  
4. Obs window ≥ 128  
5. All-ones `attention_mask` → treat as `None` (broke mid-depth)  

---

## Invention claim (current, honest)

**Mechanistic (H1+H2):** Near-lossless single-fact retrieval under KV budget requires **critical spans ± R\*=1**; at fixed bytes, **priority retention** beats equal/2× volume of non-critical tokens. Int8 on the priority set still hits ε=0 on this suite.

**Systems:** Non-oracle **seed_valley@176** matches full mid-needle through **8k** with **~27 MB** decode KV (~**42×** vs full @8k). Chunked prefill unlocks interactive 8k on 3090 WDDM. Scorer tax vs oracle remains ~176 vs 155 at 4k.

Not yet: streaming eviction (cut *peak* VRAM), oracle-matched detector, multi-hop / multi-needle, multi-model, full 3-depth L_ε curve above 4k.

---

## Next (optional)

1. **Streaming / online compress during prefill** — lower peak VRAM, push L beyond 8–12k  
2. Full 3-depth L_ε confirm at 6k/8k  
3. Multi-needle / multi-hop kill (H3)  
4. Close scorer tax 176→155  
5. int8-packed decode path; second model later
