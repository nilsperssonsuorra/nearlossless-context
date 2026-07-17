# Findings (workstation-safe: max 4k context)

**Machine:** RTX 3090 24 GB (WDDM — avoid 8k+ full KV)  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`  
**Policy:** ctx ≤ 4096 for interactive full-KV jobs  

**Project goal:** maximize \(L\) at ε≈0 quality under 24 GB (see root README + `RESEARCH_BRIEF.md`).

---

## Multi-seed paper rigor (`paper_rigor_20260716T120146Z` + stream sweep)

**Protocol:** 5 seeds × depths {0, 0.5, 1.0} = **15 cells**; seeded filler variants (`build_needle_prompt(..., seed=)`); primary Qwen3-4B @4k.

### A) H1 mechanism — **locked**

| Arm | Success | Rate |
|-----|---------|------|
| full | 15/15 | **100%** |
| oracle_r1 (crit±1) | 15/15 | **100%** |
| anti_oracle | 0/15 | **0%** |

**VERDICT: `H1_MULTI_SEED_OK`** — necessity + sufficiency hold under hay variation. Mean |oracle| ≈ **149** tokens (155 mid/start, 136 end).

### B) Scorer tax (seed_valley vs oracle)

| Metric | Value |
|--------|-------|
| Mean min budget with ε=0 (per cell) | **172** |
| Min budget with ε=0 on **all** 15 cells | **192** |
| Mean tax \(B_{\min}/\|K_{\mathrm{oracle}}\|\) | **~1.16×** (global-all **~1.24×** at 192) |
| @176 mean Jaccard vs oracle keep | **0.83** |
| @176 mean tokens kept outside oracle | **~29** |

Depth **0.0** is hardest for the scorer (often needs 192); depth 1.0 often ok at 155 (fact already near recent).

### C) Stream multi-seed (corrects earlier single-hay optimism)

| Stream budget | Rate (15 cells) | By depth (0 / 0.5 / 1.0) |
|---------------|-----------------|---------------------------|
| 512 | **33%** (5/15) | 0 / 0 / 1.0 |
| 768 | 40% | 0 / 0.2 / 1.0 |
| 1024 | 67% | 0.2 / 0.8 / 1.0 |
| **1536** | **93%** (14/15) | 0.8 / 1.0 / 1.0 |

Earlier FINDINGS “stream@512 all depths @4k” used **fixed filler** (no seed). Under multi-seed hay, **stream@1536** is the honest ε≈0 operating point at 4k. Long-\(L\) stream@1536–2048 results remain the systems story for \(L_\varepsilon\).

**Paper draft:** `papers/PAPER_DRAFT.md` · bench: `experiments/bench_paper_rigor.py`

---

## Fact capsules (new direction, 2026-07-16)

**Idea:** compress *atomic neighborhoods* (H1′ objects) under **query-unknown** streaming; sticky registry + **pin-on-exit** / **pin_hist**.

| stream budget | valley | capsules v0 sticky | capsules v1 pin-on-exit |
|---------------|--------|--------------------|-------------------------|
| 512 | 33% | 33% | 33% |
| 1024 | **67%** | 33% | 33% |
| 1536 | **93%** | 87% | 87% |

**Scored capsules v0–v1:** `CAPSULES_NO_GAIN` vs valley (packing/pin-on-exit not enough).

### Oracle-online upper bound (`capsules_20260716T162035Z`)

Perfect discovery stream: always keep critical±R if still in cache + sinks + recent.

| Arm @4k multi-seed 5×3 | @512 | @1024 |
|------------------------|------|-------|
| stream_valley | 33% | 67% |
| **stream_oracle_pin** | **100% (15/15)** | **100% (15/15)** |

**VERDICT: `DISCOVERY_IS_THE_GAP`**

- Peak budget **@512 is sufficient** if neighborhoods are known.  
- Failures of valley/capsules are **detector failures**, not “stream can’t work.”  

### Surface novelty detector (`capsules_20260716T164126Z`)

Query-unknown discovery via rarity + digit/ID-like surface features (no attention, no question):

| Arm @4k multi-seed 5×3 | @512 | @1024 |
|------------------------|------|-------|
| stream_valley | 33% | 67% |
| stream_oracle_pin | 100% | 100% |
| **stream_novelty** | **93% (14/15)** | **93% (14/15)** |

**VERDICT: `NOVELTY_DETECTOR_OK` (v0)** — closes most of the discovery gap on code needles at stream@512.

### Novelty stress suite (`novelty_stress_20260716T170938Z` + sticky recheck `…T214326Z`)

stream@512, 5 seeds (depths {0,0.5,1} except multi3 mid-only):

| Scenario | valley | **novelty** | oracle_pin | Notes |
|----------|--------|-------------|------------|-------|
| code (control) | 33% | **100%** | 100% | sticky recheck 15/15 |
| **nl** (Seraphine / Reykjavik) | 40% | **100%** (was 87% pre-sticky) | 100% | sticky recheck 15/15 |
| **prose** (soft English, no digits/IDs) | 53% | **100% (15/15)** | 100% | `…T203709Z` — rarity still separates fact from prose hay |
| **multidoc** (6 titled docs; one holds fact) | **33%** | **100% (15/15)** | **33%**† | `…T205257Z` — out-of-suite-ish multi-doc retrieval |
| **adv** (ID-like filler flood) | 33% | **100%** | 100% | pre-sticky |

† Multidoc oracle_pin@512 only passes **depth=0** (5/15): packing under budget when the relevant doc is mid/end is fragile; **novelty still 15/15**. Valley only end-depth (5/15).
| **multi3** (3 secrets recall_all) | **0/5** | **5/5 (100%)**‡ | 2/5 @512 / 5/5 @1024 | sticky + `max_new≥96` `…T202222Z` |
| **hop2** (Alice→id→password) | 1/5 @512 / 5/5 @1024 | **5/5 @512** | — | sticky |
| **hop3** (3-link + distractors) | **0/5** @512 / 5/5 @1024 | **5/5 @512** | — | sticky `…T202346Z` |

‡ Earlier multi3 “4/5” at `max_new=64` was a **decode-length** miss (seed1 listed 5/6 keys then stopped). Offline novelty covered all 6 keys; with `max_new=96` → **5/5**. Discovery was fine.

† Oracle_pin@512 multi3 packing can drop a span (2/5); @1024 = 5/5. Novelty@512 matches oracle@1024 quality on multi3 when decode budget is adequate.

**VERDICT: `NOVELTY_STRESS_PASS` + `MULTI_HOP_NOVELTY_OK` + `MULTI3_NOVELTY_OK`**

- Single-fact NL/code/adv: sticky novelty@512 solid.  
- **2-hop / 3-hop+distractors:** novelty@512 **5/5**; valley fails hard @512 (0–1/5), recovers @1024.  
- **Multi3:** novelty@512 **5/5** with sufficient `max_new`; valley **0/5**.

### Systems resources (`systems_resources_20260717T130143Z`)

Mid-depth needle, seed 0; full chunked prefill vs sticky novelty stream@512. Peak VRAM = `torch.cuda.max_memory_allocated` after prefill (includes weights).

| L | Arm | Prefill | Peak cache | Decode KV | Peak VRAM |
|---|-----|---------|------------|-----------|-----------|
| 4k | full | 1.9 s | 4046 | 569 MB | 8.5 GB |
| 4k | **novelty@512** | **1.6 s** | **1024** | **72 MB** | **8.3 GB** |
| 16k | full | 9.2 s | 16330 | 2296 MB | 10.6 GB |
| 16k | **novelty@512** | **5.9 s** | **1024** | **72 MB** | **8.3 GB** |
| 40k | full | **1237 s**† | 40916 | 5754 MB | 14.5 GB |
| 40k | **novelty@512** | **17.6 s** | **1024** | **72 MB** | **8.3 GB** |

† Full 40k chunked prefill was pathologically slow on this WDDM/3090 session (likely paging); still a valid “full path pain” data point. Novelty stays **flat ~8.3 GB** peak VRAM and **72 MB** decode KV at all three lengths.

**VERDICT: `SYSTEMS_FLAT_PEAK`** — sticky novelty stream@512 keeps peak cache/KV/VRAM essentially **independent of \(L\)** while quality holds on the mid needle; full KV grows in all three metrics.

`prefill_auto(..., mode="stream", discovery="novelty")` is now the default stream path.

Docs: `papers/CAPSULES.md` · `experiments/novelty_detect.py` · `bench_novelty_stress.py`.

### Long-L multi-seed novelty (`novelty_longL_20260716T180614Z` + sticky `…T210746Z`)

**Protocol:** 3 seeds × 3 depths = **9 cells** per L.

**v0 (no sticky)** L ∈ {8k, 12k, 16k} valley vs novelty:

| L | valley@512 | **novelty@512** | valley@1536 | novelty@1536 |
|---|------------|-----------------|-------------|--------------|
| **8k** | 33% (end only) | **100% (9/9)** | 89% | **100%** |
| **12k** | 33% | **78% (7/9)** | 78% | **89%** |
| **16k** | 33% | **100% (9/9)** | 78% | **100%** |

**v0 @24k (partial kill, same code):** novelty@512 **~67% (6/9)** — mid-depth thrash as L grows (re-rank drops early secrets).

**Sticky novelty v1** (`novelty_longL_20260716T210746Z`): sticky pin registry + score-ranked pin packing + max_capsules scales with L.

| L | novelty@512 | novelty@1536/768 | Peak cache @512 |
|---|-------------|------------------|-----------------|
| **16k** | **9/9 (100%)** | 9/9 @1536 | ~1024 |
| **24k** | **9/9 (100%)** | 9/9 @1536 | ~1024 |
| **32k** | **9/9 (100%)** | 9/9 @1536 | ~1024 |
| **40k** (`…T213723Z`) | **9/9 (100%)** | 9/9 @768 | ~1024 |

**VERDICT: `NOVELTY_LONG_L_WIN` + `STICKY_NOVELTY_FIX` + `L_EPS_40K_STICKY`**

- Under multi-seed hay, **attn valley@512 does not scale** with L (stays ~33%, end-depth only).  
- **Surface novelty@512** is multi-seed solid through **40k** once pins are sticky (24k non-sticky: 6/9 → sticky **9/9**).  
- **Systems:** multi-seed near-lossless needle through **≥40k** with peak cache **~1k tokens** (~½ prior stream@1536; ~**40×** below full @40k token count).  
- Matches prior valley fixed-filler “observed max ≥40k @1536” **at ⅓ the stream budget** under multi-seed sticky novelty.

Bench: `experiments/bench_novelty_longL.py` · detector: `novelty_detect.py` (`sticky=True` default).

### Hybrid transfer: Gemma-4 novelty (`novelty_longL_20260716T181508Z`)

Model: `google/gemma-4-E4B-it` · multi-seed 3×3 @4k · hybrid-safe compress.

| Arm | Rate |
|-----|------|
| stream_valley@512 | **33%** (end only) |
| **stream_novelty@512** | **100% (9/9)** |
| stream_valley@1024 | **100%** |
| stream_novelty@1024 | **100%** |

**VERDICT: `NOVELTY_TRANSFER_GEMMA4_OK`** — surface novelty closes the discovery gap on hybrid Gemma-4 at the same stream@512 peak class as primary Qwen3. Prior Gemma stream floor (1024 R≥2) was **attn-path** needs; with `discovery="novelty"` **stream@512** is multi-seed solid @4k.

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

### Hybrid out-of-family: Gemma-4 E4B-it (`transfer_20260715T225225Z` + retune)

Model: `google/gemma-4-E4B-it` — **hybrid** sliding-window (W=512, ~5:1) + full-attention layers; shared-KV tail.

| Arm @4k mid | Result |
|-------------|--------|
| full | **ok** |
| oracle_r1 (crit±1 on **full layers only**) | **ok**, recall=1.0, ~175 full-KV tokens |
| anti_oracle | **fail**, recall=0.0 |
| posthoc seed_valley **@176** | **ok** (after score-pass mask fix; recall≈0.87) |
| stream@512 R=1 | fail (partial / corrupted codes) |
| stream@**1024 R=2** | **ok** |
| stream@1536 R=1 | **ok** |

**Infrastructure (hybrid-safe KV):**
- Compress **full layers only**; leave sliding intact (`cumulative_length` / masks).
- `clone_dynamic_cache` preserves `DynamicSlidingWindowLayer`.
- `cache_seq_len` = full-layer length.

**Score-pass bug (fixed):** HF `create_causal_mask` uses `q_offset = past.get_seq_length()` → **layer 0** (sliding). Cropping score-cache had set sliding `cumulative_length` to local key length → full-attn masks treated queries as starting near 0 → attention mass only on **0…510**. Fix: keep sliding `cumulative_length = prefix_len` (absolute) during `crop_cache_prefix`; do not shrink sliding K/V for the score clone.

**VERDICT: `TRANSFER_OK` (hybrid; budgets model-specific)**

- H1 critical±R\* transfers to Gemma-4 **full** layers.  
- Posthoc tax matches primary (**176**) once score-pass offset is correct.  
- Stream needs **more** than Qwen/Llama defaults: **1024 + R=2** (or 1536 + R=1) @4k mid.  
- Sliding stack is a free local window; long-range budget is the full-attn layers only.

### Per-model adaptive calibration (2026-07-16)

`adaptive.policy_for(..., model_id=)` + `prefill_auto` (reads `model.config._name_or_path`):

| Family | Detection | Single posthoc floor | Single stream @4k |
|--------|-----------|----------------------|-------------------|
| primary (Qwen3-4B) | `qwen3` / default | 176 / multi-seed 192 | novelty **512** |
| qwen25 | `qwen2.5` | **320** | 512; **768** if L≥8k |
| llama32 | `llama-3.2` | **256** | 512 |
| gemma4 | `gemma-4` | 176 | novelty **512** (valley ~**1024**) |

Gemma: novelty multi-seed 9/9 @512; `prefill_auto` default discovery=novelty uses stream@512.

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
