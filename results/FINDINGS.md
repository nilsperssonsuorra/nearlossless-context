# Findings (workstation-safe: max 4k context)

**Machine:** RTX 3090 24 GB (WDDM — avoid 8k+ full KV)  
**Model:** `Qwen/Qwen3-4B-Instruct-2507`  
**Policy:** ctx ≤ 4096

---

## Headline result (`equal_byte_20260713T212102Z.csv`)

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

> **ByteBudgetKV** matches SnapKV needle quality on 4k / 3 depths while cutting **logical** KV size ~2× at equal token budget (int8), or holding ~2× tokens at equal logical bytes.

Not yet: custom int8 attention kernels, multi-model transfer, >4k without thrashing this desktop.

---

## Next (optional)

1. int8-packed decode path (real bandwidth win)  
2. Layer-pyramid byte schedules  
3. Chunked prefill so peak VRAM drops too  
4. Second model only after more 4B science  
