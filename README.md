# nearlossless-context

**Raise usable local LLM context under fixed VRAM with near–full-KV quality (ε → 0).**

Private research lab. Not a product. Not “slightly better SnapKV.”

**Read the draft:** [`papers/main.pdf`](papers/main.pdf) · figure: [`papers/figures/fig1_story.png`](papers/figures/fig1_story.png) · findings: [`results/FINDINGS.md`](results/FINDINGS.md)

---

## Results (headline)

On multi-seed retrieval (primary `Qwen3-4B`, RTX 3090):

| Result | Takeaway |
|--------|----------|
| **H1′** full / oracle crit±1 / anti-oracle | **15/15 / 15/15 / 0/15** — critical span + local radius is necessary & sufficient |
| Stream **attention** @512 multi-seed | **~33%** (end-depth only) |
| Stream **oracle pin** @512 | **15/15** — failures are **discovery**, not peak budget |
| Stream **sticky novelty** @512 | **~93–100%** @4k; **9/9 multi-seed through 40k**; peak cache **~1k** |
| multi3 / hop2 / hop3 | novelty **5/5 @512**; valley multi3 & hop3 **0/5** @512 |
| multidoc (6 titled docs) | novelty **15/15 @512**; valley/oracle_pin **5/15** |
| External-style slice (10 mixed QA @~4k) | novelty **10/10** hits = full; valley **0/10** |
| Public LongBench (60 @4k truncate) | full F1 **0.28**; novelty **~0.18** (honest gap vs suite) |
| Posthoc LB upper bound (same 60) | **0.95–1.0×** full F1 @ final 512–2048 (peak=\(L\)) |
| query_hold Pareto on LB | best **~0.92×** full F1 @ peak **~2.5k** (h2048→f1024) |
| Peak resources (novelty@512) | decode KV **~72 MB**, peak VRAM **~8.3 GB** **flat** 4k→40k |
| Transfer | H1 holds on Qwen2.5 / Llama-3.2; Gemma-4 hybrid novelty@512 **9/9** |

**Default path:** `prefill_auto(..., mode="stream", discovery="novelty")` — see `USAGE.md`.

### What we do *not* claim

General long-context SOTA, full RULER/LongBench, production int8 kernels, or that surface novelty is universal “importance.” Suite is retrieval/needle-class; see paper §Limitations.

---

## Goal

Run a small open model (**~4B**) with **much more context** than naive inference allows, on **consumer hardware**, with **little or no quality loss** vs full key–value (KV) cache.

Formally (see `RESEARCH_BRIEF.md`):

> Maximize \(L_\varepsilon\) s.t. quality ≥ \((1-\varepsilon)\times\) full-KV on suite \(\mathcal{S}\), under **≤24 GB** and usable speed.

- **ε → 0** on retrieval-critical tasks (exact facts).  
- **Theory + falsification first** — not “stack known KV tricks.”  

### Story in one line

Near-lossless training-free compression ≈ **keep critical local neighborhoods**; online stream fails when **query-unknown discovery** is weak; **sticky surface novelty** closes most of that gap on this suite through **40k** at flat peak cache; on public LongBench, **posthoc ≈ full** while online quality is a **peak/quality Pareto** (query_hold ~0.92× @ ~2.5k peak).

---

## Hardware & model

| | |
|--|--|
| GPU | NVIDIA RTX 3090 **24 GB** (Windows WDDM) |
| System RAM | 32 GB |
| Primary model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — dense full-attention GQA |
| Transfer (hybrid) | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) — full layers compress; novelty stream@512 multi-seed ok |
| Avoid (for classic KV work) | `Qwen3.5-4B` hybrid linear+full — different game |

**Workstation policy:** long \(L\) uses **streaming** (peak cache ~ budget+chunk). Full 40k prefill can thrash WDDM; prefer novelty stream for long jobs.

---

## Setup

```powershell
cd path\to\nearlossless-context   # or this folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
# CUDA torch: https://pytorch.org/get-started/locally/ if the default wheel is CPU-only
```

---

## Experiments

| Script | Purpose |
|--------|---------|
| `experiments/novelty_detect.py` | **Sticky surface-novelty** stream discovery (default) |
| `experiments/bench_novelty_stress.py` | code/nl/prose/adv/multi3/hop2/hop3 multi-seed stress |
| `experiments/bench_novelty_longL.py` | Long-\(L\) multi-seed novelty vs valley |
| `experiments/bench_systems_resources.py` | Peak VRAM / prefill time / decode KV table |
| `experiments/bench_paper_rigor.py` | Multi-seed H1 + scorer tax |
| `experiments/bench_h1_oracle.py` | **H1 kill** (oracle / anti-oracle spans) |
| `experiments/bench_h1_radius.py` | **H1′** minimum local radius \(R\) |
| `experiments/bench_h2_bytes.py` | **H2** equal-byte priority vs volume |
| `experiments/bench_h3_multi.py` / `hop.py` / `hop3.py` | Multi-needle / multi-hop arms |
| `experiments/compress_adaptive.py` | `prefill_auto` (posthoc/stream, discovery=novelty\|attn) |
| `experiments/adaptive.py` | Measured R/budget schedule + model floors |
| `experiments/bench_transfer.py` | Cross-model transfer smoke |
| `experiments/plot_paper_figures.py` | Regenerates `papers/figures/fig1_story.*` |
| `papers/build_pdf.ps1` | Builds `papers/main.pdf` from `main.tex` |

### H1 / H1′ / H2 (theory track)

```powershell
python experiments\bench_h1_oracle.py --ctx 4096 --depths 0.0,0.5,1.0
python experiments\bench_h1_radius.py --ctx 4096 --radii 0,1,2,4,8,16
python experiments\bench_h2_bytes.py --ctx 4096 --depths 0.0,0.5,1.0 --R 1
python experiments\bench_scorer_budget.py --ctx 4096 --budgets 168,192,256,384,512 --R 1
python experiments\bench_l_epsilon.py --lengths 4096,6144,8192 --allow-long --mid-only --budgets 176,256
python experiments\bench_streaming.py --lengths 4096,8192 --allow-long --budget 176 --stream-hi 512
python experiments\bench_h3_multi.py --ctx 4096 --n-needles 3 --budget 1536 --stream-budget 2048
python experiments\bench_h3_hop.py --ctx 4096 --budget 512
python experiments\bench_h3_hop3.py --ctx 4096
python experiments\bench_adaptive_e2e.py --ctx 4096
python experiments\bench_transfer.py --also-8k
python experiments\bench_transfer.py --model meta-llama/Llama-3.2-3B-Instruct --also-8k
```

Latest: stream **≥40k** primary; **H1 on Llama-3.2-3B + Qwen2.5-3B** (stream@512 ok on Llama 4k/8k; posthoc ~256–320).

### Ceiling map

```powershell
python experiments\bench_ceiling.py
# optional longer lengths (can lag the machine):
python experiments\bench_ceiling.py --lengths 2048,4096,6144 --allow-long
```

### Other (≤4k)

```powershell
python experiments\bench_context_tax.py --ctx 2048,4096
python experiments\bench_needle.py --ctx 4096 --depths 0.0,0.5,1.0
python experiments\bench_equal_byte.py --ctx 4096 --budgets 512,1024,1536
```

Outputs: `results/*.csv` + `*.json` (gitignored). Narrative: `results/FINDINGS.md`.

---

## Status

Core arc is **measured and written up** (mechanism → discovery gap → sticky novelty → long-\(L\) + multi-hop + systems table). See headline results above and `papers/main.pdf`.

**Next (optional):** arXiv submit; workshop CFP; optional SnapKV baseline arm / methods expansion — not more needle grids.

**Portfolio one-pager + paste blurb:** [`PORTFOLIO.md`](PORTFOLIO.md)

---

## Docs

| File | Content |
|------|---------|
| **`papers/main.pdf`** | **Readable paper draft (start here)** |
| `papers/PAPER_DRAFT.md` | Markdown twin of the draft |
| `papers/figures/fig1_story.png` | H1 / discovery / long-\(L\) / peak-cache figure |
| `USAGE.md` | `prefill_auto` + length guide |
| `RESEARCH_SUMMARY.md` | Short hire-aligned summary |
| `results/FINDINGS.md` | Full experimental tables / verdicts |
| `RESEARCH_BRIEF.md` | North-star brief |
| `papers/CAPSULES.md` | Fact-capsule / novelty research notes |

Related work is **crowded** (eviction + KV quant). Success here is **raising \(L\) at ε≈0 with honest multi-seed claims**, not renaming SnapKV.

---

## Repo

Private: [nearlossless-context](https://github.com/nilsperssonsuorra/nearlossless-context)

```text
nearlossless-context/
  experiments/     # benches + novelty + adaptive API
  papers/          # main.tex / main.pdf / figures / notes
  results/         # FINDINGS.md (+ local CSVs gitignored)
  USAGE.md
  RESEARCH_SUMMARY.md
  requirements.txt
```
