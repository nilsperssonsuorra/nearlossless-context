# nearlossless-context — research brief

**Repo:** private `nearlossless-context`  
**Lab:** RTX 3090 24 GB + 32 GB RAM · primary model `Qwen/Qwen3-4B-Instruct-2507`  
**Updated:** 2026-07-13  

This document is the **north star**. Implementation details live in `README.md` and `results/FINDINGS.md`. Related work notes live in `papers/NOTES.md`.

---

## 1. Problem (why anyone should care)

Long context is expensive: KV memory and bandwidth dominate decode cost as \(L\) grows; prefill cost grows with prompt length. Frontier and local serving both pay this tax.

**Product-shaped goal (lab):** run a small model with **much larger usable \(L\)** at **near–full-KV quality**.  
**Science-shaped goal:** find a **mechanism or law** that makes long memory cheap **without silent quality loss** — something that would still matter at larger scale, not only on a 4B + 3090.

Hiring is **not** guaranteed by this. Strong theory + ruthless experiments on a problem labs actually pay for is **aligned** with how research engineers get interviews. That is the bar we optimize for.

---

## 2. One success metric (hire-aligned)

Define:

- \(\mathrm{Full}(L)\): full KV (bf16/fp16) at length \(L\) — **gold quality**  
- \(\mathrm{Q}(M, L)\): quality of method \(M\) at length \(L\) on suite \(\mathcal{S}\)  
- \(R(M, L)\): resource cost (prefer **peak KV bytes** and/or **decode bytes/step**; also report peak VRAM and tok/s)  
- \(\varepsilon\): allowed relative quality gap (start with **ε = 0 on retrieval-critical tasks**, and ε ≤ 0.02 on softer tasks once suite expands)

**Primary metric:**

\[
\text{Success} = \text{maximize } L_{\varepsilon}
\quad\text{where}\quad
L_{\varepsilon} = \max\{ L : \mathrm{Q}(M,L) \ge (1-\varepsilon)\,\mathrm{Q}(\mathrm{Full},L) \}
\]
under hardware constraint (≤24 GB, usable speed).

**Secondary metric (same quality, less cost):**

\[
\text{compress}_\varepsilon(L) = \frac{R(\mathrm{Full},L)}{R(M,L)}
\quad\text{s.t.}\quad
\mathrm{Q}(M,L) \ge (1-\varepsilon)\,\mathrm{Q}(\mathrm{Full},L).
\]

**We only celebrate methods that raise \(L_\varepsilon\) a lot or raise \(\mathrm{compress}_\varepsilon\) a lot.**  
Beating “recent window” by a little at fixed 4k is **not** success.

### Suite \(\mathcal{S}\) (v0 → v1)

| Tier | Task | ε target |
|------|------|----------|
| v0 | Single needle, depths {0, 0.5, 1.0} | **0** (exact codes) |
| v0 | Mid-depth single needle under compression | **0** |
| v1 | **Multi-needle** (2–3 secrets, ask one or all) | **0** |
| v1 | **Oracle / anti-oracle** retention (below) | mechanism test |
| later | Multi-hop over long filler; agent-ish state | ε small |

Workstation: interactive full-KV jobs prefer **\(L \le 4\mathrm{k}\)** unless chunked prefill / `--allow-long`.

---

## 3. One hypothesis (H1)

### H1 — Critical-span retention is necessary; bare spans are not sufficient

**Statement (revised after kill experiment 2026-07-13):**

> For single-fact retrieval in long prompts (needle class), under training-free position compression:  
> **(Necessity)** If the cache drops the tokens encoding the fact, recovery fails (ε = 0).  
> **(Sufficiency — revised)** Retaining only the minimal fact tokens + sinks + recent window is **not** enough: the model often corrupts the fact (e.g. `maple-quartz-19` → `…-199`).  
> **(Sufficiency — H1′)** Retaining **critical tokens ± local context radius R** (plus sinks + recent/question window) restores full-KV success on this suite, at ~**25× smaller** KV than full (~20–30 MB vs ~570 MB at 4k).

**Smoke result:** `results/h1_oracle_20260713T214610Z.json` → verdict **`H1_NEEDS_LOCAL_CONTEXT`**.

**Radius sweep (H1′):** `experiments/bench_h1_radius.py` → **\(R^* = 1\)** token of local context around critical spans restores ε=0 on depths {0, 0.5, 1.0} at \(L≈4\mathrm{k}\). \(R=0\) fails on start/mid (hallucinated trailing digit). See `results/h1_radius_*.json`.

**Why this could matter at frontier scale:**  
If true, cheap long context is **not** “store a bit less of everything”; it is **guarantee critical spans + question/recent + sinks**, and spend remaining bytes optimally. That drives detectors, controllers, and training objectives — not another generic top-k.

**What H1 is *not*:**  
A claim that SnapKV is new, or that int8 is new. Those are tools for testing H1.

### Implications if H1 is true

1. Metrics should track **critical-span recall** in the cache, not only output accuracy.  
2. Methods should be judged by **bytes per retained critical bit**, not tokens alone.  
3. Multi-hop / agent state will need a **stronger** hypothesis later (structure, not just spans).

### Implications if H1 is false

- **Necessity fails:** model recovers fact without span in cache → leakage via other tokens / weights / prompt artifacts; suite is invalid.  
- **Sufficiency fails:** spans retained but answer wrong → RoPE/layout/precision/decode bugs or need for more than spans (context around fact, precision on keys, etc.). That failure is also a discovery if characterized.

---

## 4. Kill experiment (falsification protocol)

**Script:** `experiments/bench_h1_oracle.py`  

**Fixed:** model = primary 4B, \(L \approx 4\mathrm{k}\), mid-depth fact(s), same decode policy as other benches, no all-ones mask footgun.

| Arm | Cache contents | Predict if H1 true |
|-----|----------------|--------------------|
| **Full** | All tokens | Success (gold) |
| **Oracle-keep** | sinks + **critical span(s)** + recent/question window only | **Success ≈ Full** |
| **Anti-oracle** | same *budget* as oracle, but **exclude** critical span; fill with high-attention or random other tokens | **Failure** (cannot get fact) |
| **Recent-only** | last \(W\) tokens only (control) | Fail unless fact is at end |
| **SnapKV / ByteBudget** | our methods | Should succeed only if they retain critical spans |

**Kill rules:**

1. **Kill sufficiency:** Oracle-keep success rate ≪ Full on ≥ N trials (different fillers/depths) → H1 sufficiency false.  
2. **Kill necessity:** Anti-oracle success rate ≈ Full → H1 necessity false (or task broken).  
3. **Support H1:** Oracle ≈ Full **and** Anti-oracle ≈ 0 **and** method success correlates with measured span-in-cache.

**Minimum N:** 3 depths × ≥1 prompt seed for smoke; then ≥5 seeds for a real claim.

**Logged every run:** span token indices, whether each arm retained them, exact match on secrets, cache token count, KV bytes.

---

## 5. What we do *after* H1 (only one branch)

| Result | Next |
|--------|------|
| H1 supported | H2: at **fixed bytes**, precision on **keys** of critical spans vs more irrelevant tokens — equal-byte Pareto aimed at raising \(L_\varepsilon\) |
| Sufficiency fails | Diagnose RoPE/precision/layout; H1' about **representation fidelity**, not just positions |
| Necessity fails | Rebuild suite until gold requires the span |
| H1 only works for single needle | H3: multi-hop needs **relational** retention (edges, not entities) |

**Status (2026-07-15):** H1–H3 green; stream **≥40k** primary; **H1 transfers to Qwen2.5-3B and Llama-3.2-3B** (budgets model-specific). **Next:** residual tax; per-model calibration; writeup.

Do **not** add new methods until the kill experiment is green or H1 is revised.

---

## 6. Lab constraints (non-negotiable)

| | |
|--|--|
| Primary model | `Qwen/Qwen3-4B-Instruct-2507` (full attention) |
| One model until H1 settled | No multi-model zoo |
| Interactive length | Prefer \(L \le 4096\) full KV |
| Gold baseline | Full KV always reported |
| No success theater | Report anti-oracle and failures |

---

## 7. Related work (one paragraph)

Training-free KV eviction (StreamingLLM, H2O, SnapKV, PyramidKV, Ada-KV, RocketKV, …) and KV quantization (KIVI, FP8/INT cache, …) already make context cheaper. **None of that is our discovery claim.** We use them as baselines and tools. Our claim is about **what must be retained for near-lossless retrieval under a budget** (H1: critical±R\*), then **how to spend remaining bytes** (H2: priority beats volume) to raise \(L_\varepsilon\).

---

## 8. Career note (honest)

Making long context cheaper is a **real** lab problem. Strong public or private evidence of a **new mechanism + falsifiable results** can help get interviews. It does **not** “definitely” produce an OpenAI offer. Optimize for **truth and clarity**, not for fantasy guarantees.

---

## 9. Immediate next action

```powershell
python experiments\bench_h1_oracle.py --ctx 4096 --depths 0.0,0.5,1.0
```

Pass/fail H1 on the printed summary; update `results/FINDINGS.md` with the kill outcome before any new method work.
