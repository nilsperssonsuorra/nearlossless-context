# Paper cards (read 2026-07-13)

## Model ages (read this first)

| | Papers cite | **We run** |
|--|-------------|------------|
| Typical | Llama-2, Mistral-7B-v0.2, Llama-3.1-8B | **Qwen3.6** (27B / 35B-A3B), **Qwen3.5** smalls, **Gemma 4**, Mistral Small 3.x |
| Why old names appear | Papers publish on models available *then* | Our invention must work on **current** local models |
| Rule | Never treat paper’s model list as our experiment list | Re-eval on **Qwen3.6 / Gemma 4** — not Qwen3-only, not Gemma 3 |

---

## SnapKV (2404.14469)

**Problem:** Long *prompts* blow up KV; prior eviction focused more on decode steps than prompt.  
**Key finding:** Important prefix tokens for generation are predictable from an end-of-prompt **observation window**; stable across generation.  
**Algo:** Per-head attention vote from obs queries → pool1d clustering → keep top-k prefix + full obs window.  
**Numbers:** 3.6× gen speed, 8.2× mem @ 16k; 380k ctx on A100-80GB with tiny cache; LongBench ≈ full.  
**Weakness:** Permanent drop; multi-query importance shifts; uniform budget per head.  
**Baseline priority:** P0

## PyramidKV (2406.02069)

**Problem:** Uniform KV size across layers ignores attention structure.  
**Key finding:** **Pyramidal information funneling** — lower layers broad, upper layers sparse massive attention.  
**Algo:** More cache lower layers, less upper (arithmetic sequence); SnapKV-like selection within layer.  
**Numbers:** ~12% KV ≈ full LongBench; extreme 0.7% still useful; NIAH strong.  
**Weakness:** Hand-tuned pyramid shape β; heads within layer still uniform in base method.  
**Baseline priority:** P0

## Ada-KV (2407.11550)

**Problem:** Uniform **head** budgets ignore sparse vs dispersed heads.  
**Key finding + theory:** L1 eviction-loss upper bound; Top-k optimal for fixed Bi; **cross-head Top-B frequency** gives optimal adaptive Bi.  
**Algo:** Ada-SnapKV / Ada-Pyramid; α safeguard; GQA mean; varlen FlashAttention storage.  
**Evals:** RULER + LongBench; **question-aware vs question-agnostic** (agnostic much harder).  
**Numbers:** Large gains in agnostic + low budget.  
**Weakness:** Token budgets only (not bit budgets); not consumer-bandwidth-primary.  
**Baseline priority:** P0 — **do not reinvent**

## KIVI (2402.02750)

**Problem:** KV quant without understanding K vs V structure fails at 2-bit.  
**Key finding:** K → per-channel quant; V → per-token quant; residual FP window helps hard tasks.  
**Numbers:** 2.6× peak mem; batch↑; throughput 2.3–3.5×.  
**Weakness:** No token eviction; older model suite in paper.  
**Baseline priority:** P0 (or llama.cpp q4/q8 KV as practical stand-in)

## RocketKV (2502.14051)

**Problem:** Permanent-only or dynamic-only miss oracle Top-k at low budgets.  
**Key finding:** Unique top-k set over full decode << sequence length; two stages help.  
**Algo:** SnapKV coarse permanent → Hybrid Sparse Attention (seq pages + head dim); adaptive ratio split; MT variant for multi-turn.  
**Numbers:** up to 400× compression, 3.7× decode e2e A100, ~33% peak mem save.  
**Note:** Authors expect **larger** speedups on consumer GPUs (no HBM).  
**Baseline priority:** P1 (if environment allows)

## Ada-KV vs our invention

| | Ada-KV | ByteBudgetKV (ours) |
|--|--------|---------------------|
| Resource unit | tokens per head | **bytes** per head/layer |
| Precision | uniform (usually fp16/bf16 KV) | **heterogeneous bits** |
| Primary metric in papers | quality @ token budget | quality **+** tok/s @ **byte/VRAM** on **3090** |
| Layers | via Ada-Pyramid | pyramid in **bytes** |
| Novelty | already published | joint slots×bits under byte constraint |

## Code links

- SnapKV: https://github.com/FasterDecoding/SnapKV  
- PyramidKV: https://github.com/Zefan-Cai/PyramidKV  
- Ada-KV: https://github.com/FFY0/AdaKV  
- KIVI: https://github.com/jy-yuan/KIVI  
- RocketKV: https://github.com/NVlabs/RocketKV  
- kvpress (NVIDIA multi-method): https://github.com/NVIDIA/kvpress  
