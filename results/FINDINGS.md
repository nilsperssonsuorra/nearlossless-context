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

## H3 multi-needle (`h3_multi_20260715T115618Z` + budget follow-ups)

**Setup:** 3 distinct secrets at depths ≈0.17 / 0.5 / 0.83, L≈4k, task **recall_all** (6 keys) + **recall_one** (mid pair).

| Arm | recall_all | Notes |
|-----|------------|--------|
| full | **6/6** | suite valid |
| **oracle_r1** (all crit±1 + sinks + recent) | **6/6** | keep ≈**248** tokens only |
| anti_oracle | **0/6** | necessity holds |
| recent@512 | **0/6** | |
| posthoc@512 | 4/6 (recall_c≈0.79) | partial spans |
| posthoc@1024 | 4/6 | still partial |
| **posthoc@1536** | **6/6** (recall_c=1.0) | scorer tax vs oracle ≈**6×** |
| stream@512 | 3/6 | |
| stream@1536 | 5/6 | |
| **stream@2048** | **6/6** | online multi-span OK |

**recall_one** (ask mid pair only): full/oracle **ok**; posthoc@512 **ok**; stream@512 **fail**; anti **fail**.

**VERDICT: `H3_MECHANISM_HOLDS` + scorer budget scales with #spans**

- Multi-needle does **not** break H1/H2: keep **all** critical±R\* → full quality; drop them → fail.  
- Oracle multi-needle still tiny (~**248** vs ~4k).  
- Non-oracle scorers need **much larger budgets** than single-needle (posthoc **1536** vs single-needle **176**; stream **2048** vs **512**).  
- Failure mode unchanged: truncated entities (`CRIMSON-3301`, `CRIMIN-3301`).

Science takeaway: **entity-neighborhood retention generalizes to multi-needle**; the hard part is **detecting multiple disjoint critical regions** under a tight byte budget, not a new retention law.

### Multi-needle scorer tax mitigation (R expand)

Tried multipeak / binned selection: **no gain** (same recall curve as vanilla valley) → problem is not seed diversity among peaks.

**Larger completion radius** on the same scorer:

| Posthoc expand R | Min budget for 6/6 | vs oracle (~248) |
|------------------|--------------------|------------------|
| R=1 | **1536** | ~6.2× |
| R=4 | still 5/6 @1280 | — |
| **R=8** | **384** | **~1.5×** |

Single-needle mid still ε=0 at **R=8, B=176**.

**Takeaway:** under imperfect multi-span detection, **forgiving local completion (larger R)** recovers entity tokens near weak peaks; smarter peak ranking alone did not. Scorer tax for 3 needles drops from **~6× → ~1.5×** oracle with R=8@384.

### Stream multi-needle (R=1 vs R=8)

| Stream expand | Min budget for 6/6 @4k | Peak cache |
|---------------|------------------------|------------|
| R=1 | **2048** | ~2560 |
| **R=8** | **1024** | **~1536** |

R=8 halves stream multi-needle budget (not as large a win as posthoc 1536→384, but real).

### Multi-hop two-fact (`h3_hop_20260715T121216Z`)

Link: Alice → E-4412 → maple-quartz-19 (facts at ~0.3 / ~0.7 depth).

| Arm | Success | Cache |
|-----|---------|-------|
| full | **yes** | ~4k |
| oracle_r1 (both facts ±1) | **yes** | **~167** |
| anti_oracle | **no** | ~156 |
| recent | no | 512 |
| posthoc R=1/8 @512 | **yes** | ~519 |
| stream R=1/8 @512 | **yes** | ~519 |
| posthoc R=8 @384 | **yes** | ~391 |

**VERDICT: `HOP_SUPPORTED`**

- Two-hop single answer still fits the **critical-span** story (both bridge facts).  
- Scorers handle this easier than 3-way recall_all (one password target).  
- Necessity holds: drop critical spans → wrong answer.

### 3-hop + distractors (`h3_hop3_20260715T121607Z`)

Alice → Dept-7 → officer E-4412 → maple-quartz-19, with Bob/Carol wrong chains (pine-nebula-88, oak-cipher-42).

| Arm | Success | Notes |
|-----|---------|--------|
| full | **yes** | |
| oracle_r1 | **yes** @~215 tok | |
| anti_oracle | **no** | necessity |
| recent | no | |
| posthoc R=1@512 / R=8@384 / adaptive | **yes** | |
| stream R=1@512 | **no** | answered **oak-cipher-42** (distractor) |
| stream R=8@1024 / adaptive | **yes** | |

**VERDICT: `HOP3_SUPPORTED`**

- Critical-span retention still necessary+sufficient with **3 links + distractors**.  
- Streaming at single-needle budgets (**512, R=1**) is **unsafe** under distractors (picks wrong chain).  
- Adaptive policy (R=8, stream 1024) from `adaptive.py` passes.

### Adaptive policy (`experiments/adaptive.py` + `compress_adaptive.py`)

Lab schedule maps (n_entities, L, multi_hop) → (R, budget, stream_budget).

**E2E** (`adaptive_e2e_20260715T122630Z`):

| Task | posthoc true-n | posthoc **auto** (peak n̂) | stream true-n | stream L-only |
|------|----------------|---------------------------|---------------|---------------|
| single mid | PASS @176 | PASS (n̂=2 → looser 384) | PASS | PASS |
| multi3 recall_all | PASS | **PASS** (n̂=2 → R=8@384) | PASS | FAIL |
| hop3+distractors | PASS | PASS | PASS | FAIL (distractor) |

**VERDICT: `ADAPTIVE_E2E_OK`**

- Peak estimate must **ignore sink mass** (else n̂ stuck at 1).  
- Auto n̂ often 2 not 3, but R=8@384 schedule still recovers 3 needles.  
- Stream still needs **entity hint or conservative multi budget** — L-only fails multi/hop3.  
- True n_entities remains best; auto is usable for **posthoc**.

## Transfer smoke

### Same family: Qwen2.5-3B (`transfer_20260715T214430Z` + retune)

| Arm @4k mid | Result |
|-------------|--------|
| full / oracle_r1 / anti | **ok / ok / fail** (H1) |
| stream@512 | **ok** @4k; **needs 768** @8k |
| posthoc | **@320** (not 176) |

### Out-of-family: Llama-3.2-3B-Instruct (`transfer_20260715T215958Z` + retune)

| Arm | Result |
|-----|--------|
| full / oracle_r1 / anti | **ok / ok / fail** (H1) |
| stream@512 | **ok @4k and @8k** |
| posthoc@176 | fail → **ok @256** |

**VERDICT: `TRANSFER_OK` (mechanism); budgets are model-specific**

- H1 critical±R\* **transfers across Qwen3-4B, Qwen2.5-3B, and Llama-3.2-3B**.  
- Streaming@512 is robust on Llama (even 8k); Qwen2.5 needs more @8k.  
- Posthoc scorer tax: primary ~176, Llama ~256, Qwen2.5 ~320.

### Hybrid out-of-family: Gemma-4 E4B-it (`transfer_20260715T223926Z` + debug)

Model: `google/gemma-4-E4B-it` — **hybrid** sliding-window (W=512, ~5:1) + full-attention layers; shared-KV tail. Not a pure full-attn target.

| Arm @4k mid | Result |
|-------------|--------|
| full | **ok** (suite valid) |
| oracle_r1 (crit±1 on **full layers only**) | **ok**, recall=1.0, ~175 full-KV tokens |
| anti_oracle | **fail**, recall=0.0 (H1 necessity) |
| stream@512 / 768 / 1024 | **fail** (512: partial `BLUE-ORBIT` only) |
| posthoc seed_valley @176–512 | **fail**, recall=0.0 |

**Infrastructure fix (required for any hybrid compress):**
- `compress_keep_indices` / SnapKV compress **skip sliding layers** (already windowed; masks use `cumulative_length`).
- `clone_dynamic_cache` / `crop_cache_prefix` preserve `DynamicSlidingWindowLayer`.
- `cache_seq_len` reports **full-layer** length (compressible store).

**Why scorer fails:** eager score-pass attentions on full layers only put mass on indices **0…510** (nnz=511 = sliding window), never mid-context critical (~2000). Sink collapse — seed_valley cannot recover the needle. Oracle still proves H1 on full-layer KV.

**VERDICT: `TRANSFER_H1_OK_SCORER_FAIL` (hybrid)**

- H1 critical±R\* **transfers to Gemma-4 full layers**.  
- Classic attention scorer / stream path **does not** transfer without a hybrid-aware score signal (non-attn or fixed score-pass mask).  
- Sliding layers are a free local window; long-range budget is the full-attn stack only.

### Stream-time auto-raise + long-L policy (2026-07-15 evening)

| Change | Result |
|--------|--------|
| Mid-stream peak probe **before first hard drop** (`auto_raise_budget`) | multi3 stream **without** `n_entities`: **6/6** (raised to budget 1024, R=8) |
| L≥28k policy → stream **2048** | 28k mid **ok** (was flake @1536) |
| Optional **int8** final fake-quant | single posthoc@176: quality ok; logical ~**12.6 MB** vs runtime ~27 MB |

---

## Streaming prefill compress (`streaming_20260714T223948Z` + sweeps)

**Goal:** cut *peak* KV during prefill (not only final decode KV).

| Method | Peak cache | Final cache | Mid@4k/8k | 3 depths@4k/8k |
|--------|------------|-------------|-----------|----------------|
| full_chunked | =L (4k–8k) | =L | ok | (full) |
| posthoc seed_valley@176 | =L | ~192 | ok | ok @4k prior |
| **stream@512→512** | **~1024** | **512** | **ok** | **ok all** |
| stream@176→176 | ~688 | ~176 | **fail** | — |
| stream@\*→176 final tighten | high | 176 | **fail** | — |
| recent@176 | =L | ~176 | fail mid | — |

**VERDICT: `STREAM_PEAK_WIN` (at budget 512)**

- Online seed_valley after each chunk with **stream_budget=512** keeps ε=0 on **all depths {0, 0.5, 1.0}** at **4k and 8k**.  
- **Peak cache ~1024** vs full **8150** @8k (~**8×** lower peak tokens).  
- Peak VRAM ~**8.5 GB** (stable) vs posthoc valley ~**12.8 GB** (score-clone on full past).  
- Final decode KV ~**72 MB** (512 tok) vs full ~**1148 MB** @8k (~**16×**); looser than posthoc@176 (~27 MB).  
- **Cannot** currently final-tighten stream→176: intermediate compress uses pre-question queries and already thins spans; posthoc@176 still needs full past + question window.

Science takeaway: for peak-VRAM-limited \(L_\varepsilon\), **online retention at a moderate budget (512)** beats “full then compress to 176.” Query-agnostic mid-stream scoring is the bottleneck for oracle-tight online budgets.

### Long-L streaming ceiling (2026-07-15)

All depths {0, 0.5, 1.0}; peak cache ≈ budget + chunk (512).

| Stream budget | L=8k | L=12k | L=16k | Peak cache | Peak VRAM |
|---------------|------|-------|-------|------------|-----------|
| **512** | **3/3** | 2/3 (mid fail `…-qu-qu-19`) | 2/3 mid fail | ~1024 | ~8.5 GB |
| 768 / 1024 | — | mid fail | mid fail | ~1.3–1.5k | ~8.6–8.8 GB |
| **1536** | — | **3/3** | **3/3** | **~2048** | **~9.1 GB** |
| full mid only | ok | ok | ok | =L (8–16k) | 9.4–11.0 GB |

**\(L_\varepsilon\) (ε=0, all depths):**

| Method | \(L_\varepsilon\) | Peak cache @ that L | Decode KV |
|--------|-------------------|---------------------|-----------|
| stream@512 | **8192** | ~1024 | ~72 MB |
| **stream@1536** | **≥16384** (tested) | **~2048** | **~216 MB** |
| full mid | ≥16384 | ~16k | ~2300 MB |

**VERDICT: `L_EPS_RAISED` under peak-cache constraint**

- Streaming reaches **16k** quality with peak cache **~2k** (~**8×** below full @16k) and VRAM **~9.1 GB** vs full **~11.0 GB**.  
- Mid-depth is the hard depth as L grows: need **higher stream budget** (512→1536 from 8k→16k), not more recent-only.  
- Full still fits 16k on 3090 with chunked prefill, but pays **~2.3 GB** decode KV vs **~0.22 GB** stream@1536 (~**10×**).

### Practical max-L ceiling (stream, single-needle, all depths)

Extended probe 2026-07-15 (stream valley R=1, budget 1536):

| L | stream@1536 | Notes |
|---|-------------|--------|
| 16–24k | **PASS 3/3** | solid |
| 28k | mid **FAIL** once | **PASS** mid @ B=2048 |
| 32k | **PASS 3/3** | |
| **40k** | **PASS 3/3** | peak cache ~2048, VRAM ~9.1 GB |

**Comfortable / reliable:** **~24k** @1536.  
**Observed max this suite:** **≥40k** @1536 (all depths); quality edge can be **noisy** (~28k mid flaked).  
**Peak resources stay flat** (~2k cache tokens, ~9 GB) as L grows — the whole point of streaming.

vs “before” **~4k comfortable full**: about **6–10×** longer near-lossless single-needle context.

API: `prefill_auto(model, ids, mode="stream")`; multi-secret → `safe_multi=True` or `n_entities=3`.

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

**Systems:**  
- **Posthoc** seed_valley@176: ~**42×** decode KV @8k mid (full peak).  
- **Streaming @1536:** reliable **~24k**, observed **≥40k** single-needle (peak ~2k, ~9 GB; 28k mid noisy).  
- **Adaptive posthoc** + **`prefill_auto`** (`safe_multi` for multi-secret stream).  

Not yet: multi-needle at oracle keep size on all models, true int8 kernels, multi-hop transfer suite.

---

## Next (optional)

1. Residual scorer tax → oracle (primary + Llama)  
2. Per-model adaptive calibration table  
3. Public writeup
