# nearlossless-context

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21894255.svg)](https://doi.org/10.5281/zenodo.21894255)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21894719.svg)](https://doi.org/10.5281/zenodo.21894719)

**A research library for memory-efficient long-context inference under fixed VRAM.** It implements critical-span retention, query-unknown discovery, and measured KV-cache quality/peak-memory tradeoffs for Hugging Face models.

## Install

```powershell
pip install nearlossless-context
```

The distribution name is `nearlossless-context`; the import package is `nearlossless_context`. Install a CUDA-enabled PyTorch build appropriate for your system when GPU execution is required.

For development or paper reproduction, install from source:

```powershell
git clone https://github.com/nilsperssonsuorra/nearlossless-context.git
cd nearlossless-context
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Quick start

```python
from nearlossless_context import greedy_generate, prefill_auto

# Default online path: sticky surface-novelty discovery.
past, logits, info = prefill_auto(
    model,
    input_ids,
    mode="stream",
    tokenizer=tokenizer,
)

tokens = greedy_generate(
    model,
    past,
    logits,
    max_new,
    eos_id=tokenizer.eos_token_id,
    next_position=input_ids.shape[-1],
)
```

Use `discovery="query_hold"` for the measured LongBench peak/quality tradeoff, or `mode="posthoc"` when query-aware scorer quality matters more than peak prefill memory. See [`USAGE.md`](USAGE.md) for model-family floors, multi-document settings, and generation requirements.

| | |
|--|--|
| **Paper** | [`papers/main.pdf`](papers/main.pdf) · [doi:10.5281/zenodo.21894255](https://doi.org/10.5281/zenodo.21894255) |
| **Software releases** | [GitHub](https://github.com/nilsperssonsuorra/nearlossless-context/releases) · [all-version DOI: 10.5281/zenodo.21894719](https://doi.org/10.5281/zenodo.21894719) |
| **Figure** | [`papers/figures/fig1_story.png`](papers/figures/fig1_story.png) |
| **Findings** | [`results/FINDINGS.md`](results/FINDINGS.md) |
| **Usage** | [`USAGE.md`](USAGE.md) |

---

## Story in one line

Near-lossless training-free KV compression ≈ **keep critical local neighborhoods**; online stream fails when **query-unknown discovery** is weak; **sticky surface novelty** closes most of that gap on a multi-seed retrieval suite through **40k** at flat peak cache ~1k; on public LongBench, **posthoc query-aware ≈ full F1** while online quality is a **peak/quality Pareto** (query_hold ~0.92× full @ ~2.5k peak).

---

## Results (headline)

On multi-seed retrieval (primary `Qwen3-4B`, RTX 3090):

| Result | Takeaway |
|--------|----------|
| **H1′** full / oracle crit±1 / anti-oracle | **15/15 / 15/15 / 0/15** — critical span + local radius is necessary & sufficient |
| Stream **attention** @512 multi-seed | **~33%** (end-depth only) |
| Stream **oracle pin** @512 | **15/15** — failures are **discovery**, not peak budget |
| Stream **sticky novelty** @512 | **~93%** @4k surface novelty; **9/9 cells through 40k** sticky; peak cache **~1k** |
| multi3 / hop2 / hop3 | novelty **5/5 @512**; valley multi3 & hop3 **0/5** @512 |
| multidoc (6 titled docs) | novelty **15/15 @512**; valley/oracle_pin **5/15** |
| External-style slice (10 mixed QA @~4k) | novelty **10/10** hits = full; valley **0/10** |
| Public LongBench (60 @4k truncate) | full F1 **0.28**; novelty **~0.18** (honest gap vs suite) |
| Posthoc LB upper bound (same 60) | **0.95–1.0×** full F1 @ final 512–2048 (peak=\(L\)) |
| query_hold Pareto on LB | best **~0.92×** full F1 @ peak **~2.5k** (h2048→f1024) |
| Peak resources (novelty@512) | decode KV **~72 MB**, peak VRAM **~8.3 GB** **flat** 4k→40k |
| Transfer | H1 holds on Qwen2.5 / Llama-3.2; Gemma-4 hybrid novelty@512 **9/9** |

**Default path:** `prefill_auto(..., mode="stream", discovery="novelty")` — see [`USAGE.md`](USAGE.md).

### What we do *not* claim

General long-context SOTA, full RULER/LongBench leaderboards, production int8 kernels, or that surface novelty is universal “importance.” Suite is retrieval/needle-class; open long-doc QA is a Pareto, not free lunch at peak~1k. See paper §Limitations.

---

## Goal

Run a small open model (**~4B**) with **much more context** than naive inference allows, on **consumer hardware**, with **little or no quality loss** vs full key–value (KV) cache on retrieval-critical tasks.

Formally:

> Maximize \(L_\varepsilon\) s.t. quality ≥ \((1-\varepsilon)\times\) full-KV on suite \(\mathcal{S}\), under **≤24 GB** and usable speed.

- **ε → 0** on retrieval-critical tasks (exact facts).  
- **Theory + falsification first** — not “stack known KV tricks.”  

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

## Experiments

| Script | Purpose |
|--------|---------|
| `experiments/novelty_detect.py` | Sticky surface-novelty stream discovery (default) |
| `experiments/bench_external_slice.py` | Offline / LongBench slice; posthoc UB + query_hold Pareto |
| `experiments/bench_novelty_stress.py` | code/nl/prose/adv/multi3/hop2/hop3 multi-seed stress |
| `experiments/bench_novelty_longL.py` | Long-\(L\) multi-seed novelty vs valley |
| `experiments/bench_systems_resources.py` | Peak VRAM / prefill time / decode KV table |
| `experiments/bench_paper_rigor.py` | Multi-seed H1 + scorer tax |
| `experiments/bench_h1_oracle.py` | H1 kill (oracle / anti-oracle spans) |
| `experiments/compress_adaptive.py` | `prefill_auto` (posthoc/stream, discovery=novelty\|attn\|query_hold) |
| `experiments/plot_paper_figures.py` | Regenerates `papers/figures/fig1_story.*` |
| `papers/build_pdf.ps1` | Builds `papers/main.pdf` from `main.tex` |

### Paper reproduction (core)

```powershell
python experiments\bench_paper_rigor.py --seeds 0,1,2,3,4 --ctx 4096
python experiments\bench_capsules.py --novelty --oracle-online --skip-capsules --budgets 512
python experiments\bench_novelty_longL.py --ctx 16384,24576,32768,40960 --arms novelty --budgets 512
python experiments\bench_external_slice.py --source longbench --n 60 --arms full,posthoc --budgets 512,1024,2048
python experiments\plot_paper_figures.py
```

Outputs: `results/*.csv` + `*.json` (gitignored). Narrative: [`results/FINDINGS.md`](results/FINDINGS.md).

---

## Status

Core arc is **measured and written up**: mechanism → discovery gap → sticky novelty → long-\(L\) + stresses → public LongBench decomposition (posthoc UB + query_hold Pareto).

This is an **experimental alpha API** intended for research and evaluation. It is not yet a production serving backend. Interfaces and measured operating points may change as integration work expands.

---

## Roadmap

The project is moving from a reproducible research prototype toward an integration-ready long-context inference library. Unchecked items are planned work, not claims about current capabilities.

### Near term

- [ ] Validate wheel and source-distribution installation in clean environments
- [ ] Publish the package to PyPI after TestPyPI verification
- [ ] Expand unit tests and CPU-compatible public-API smoke tests
- [ ] Stabilize the compression-policy interface and runnable examples
- [ ] Add lightweight reproducibility checks to CI

### Integration work

- [ ] Prototype integration with a vLLM-compatible KV-cache or attention extension point
- [ ] Investigate an SGLang integration path
- [ ] Add real quantized KV storage and kernel support
- [ ] Extend evaluation across additional models, context lengths, and public datasets

### Longer term

- [ ] Add scheduled GPU benchmarking with regression tracking
- [ ] Evaluate batching, concurrency, and production-serving constraints
- [ ] Prepare upstream draft PRs once interfaces and benchmarks are sufficiently mature

Priorities may change as benchmark evidence and upstream interfaces evolve. Contributions and reproducible issue reports are welcome.

---

## Docs

| File | Content |
|------|---------|
| **`papers/main.pdf`** | **Readable preprint (start here)** |
| `papers/PAPER_DRAFT.md` | Markdown twin of the preprint |
| `papers/figures/fig1_story.png` | H1 / discovery / long-\(L\) / peak-cache figure |
| `USAGE.md` | `prefill_auto` + length guide |
| `results/FINDINGS.md` | Full experimental tables / verdicts |

Related work is **crowded** (eviction + KV quant). Success here is **raising \(L\) at ε≈0 with honest multi-seed claims** and a clear discovery diagnosis—not renaming SnapKV.

---

## Repo layout

```text
nearlossless-context/
  experiments/     # benches + novelty + adaptive API
  papers/          # main.tex / main.pdf / figures
  results/         # FINDINGS.md (+ local CSVs gitignored)
  USAGE.md
  requirements.txt
```

**GitHub:** [nilsperssonsuorra/nearlossless-context](https://github.com/nilsperssonsuorra/nearlossless-context)

---

## License

The software in this repository is licensed under the [Apache License 2.0](LICENSE).
The paper and figures are copyright © 2026 Nils Persson Suorra and licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Models,
datasets, and other third-party materials remain subject to their respective
licenses.
