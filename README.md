# nearlossless-context

**Raise usable local LLM context under fixed VRAM with near–full-KV quality (ε → 0).**

Private research lab. Not a product. Not “slightly better SnapKV.”

---

## Goal

Run a small open model (starting with **4B**) with **much more context** than naive inference allows, on **consumer hardware**, with **little or no quality loss** vs full key–value (KV) cache.

Formally (see `RESEARCH_BRIEF.md`):

> Maximize \(L_\varepsilon\) s.t. quality ≥ \((1-\varepsilon)\times\) full-KV on suite \(\mathcal{S}\), under **≤24 GB** and usable speed.  
> Secondary: maximize compression ratio at fixed \(L\) with the same ε constraint.

- **ε → 0** on retrieval-critical tasks (exact facts).  
- **Theory + falsification first** — not “stack known KV tricks.”  
- Career: strong evidence on a real lab problem helps; it does **not** guarantee any offer.

### What this is not

- Rebranding known eviction tricks for a paper delta  
- Infinite context with perfect dense attention on one 3090 (not realistic as pure dense KV)  
- Multi-model zoo before the 4B ceiling is understood  

### Paths we care about (in order)

1. **Dense ceiling** — how far full / near-lossless KV (e.g. quant) goes before OOM or thrash  
2. **Near-lossless compression** — only if it preserves the suite at long \(L\)  
3. **Memory hierarchy** — hot full KV + warm/cold store, only after dense is exhausted  

---

## Hardware & model

| | |
|--|--|
| GPU | NVIDIA RTX 3090 **24 GB** (Windows WDDM) |
| System RAM | 32 GB |
| Primary model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — dense full-attention GQA |
| Avoid (for classic KV work) | `Qwen3.5-4B` hybrid linear+full — different game |

**Workstation policy:** prefer **≤4k** for interactive runs. Longer lengths need chunked prefill / careful jobs so the desktop does not freeze. Use `--allow-long` only when you mean it.

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
| `experiments/bench_h1_oracle.py` | **H1 kill experiment** (oracle / anti-oracle spans) |
| `experiments/bench_h1_radius.py` | **H1′** minimum local radius \(R\) around critical spans |
| `experiments/bench_h2_bytes.py` | **H2** equal-byte: priority (crit±R\*) vs volume |
| `experiments/bench_scorer_budget.py` | Non-oracle scorer vs oracle at tight budgets |
| `experiments/bench_l_epsilon.py` | **L_ε** vs length (chunked prefill + budgeted KV) |
| `experiments/scorer_valley.py` | seed_valley selection + compress helpers |
| `experiments/bench_ceiling.py` | Dense full-KV ceiling: VRAM / speed / needle vs \(L\) |
| `experiments/bench_context_tax.py` | Decode/VRAM tax vs length |
| `experiments/bench_compare.py` | Full vs recent vs SnapKV-style (≤4k) |
| `experiments/bench_needle.py` | Needle-in-a-haystack smoke |
| `experiments/bench_equal_byte.py` | Quality vs KV budget |
| `experiments/bytebudget.py` | ByteBudget tooling (int8 logical) |

### H1 / H1′ / H2 (theory track)

```powershell
python experiments\bench_h1_oracle.py --ctx 4096 --depths 0.0,0.5,1.0
python experiments\bench_h1_radius.py --ctx 4096 --radii 0,1,2,4,8,16
python experiments\bench_h2_bytes.py --ctx 4096 --depths 0.0,0.5,1.0 --R 1
python experiments\bench_scorer_budget.py --ctx 4096 --budgets 168,192,256,384,512 --R 1
python experiments\bench_l_epsilon.py --lengths 4096,6144,8192 --allow-long --mid-only --budgets 176,256
```

Latest: **\(R^*=1\)**; **H2_SUPPORTED**; **seed_valley@176** matches full mid-needle through **8k** (~42× smaller decode KV).

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

## Status (lab so far)

- Runnable HF lab on 3090; DynamicCache + RoPE/mask footguns documented and fixed for compression paths  
- **H1 / H1′:** critical spans need local radius \(R^*=1\); bare fact tokens insufficient  
- **H2:** at fixed bytes, priority (crit±R\*) beats equal/2× volume without critical spans  
- **Scorer:** **seed_valley@176** ε=0; beats SnapKV@192 at 4k; oracle@**~155**  
- **L_ε (mid):** full and seed_valley@176 both reach **8k**; decode KV **~27 MB vs ~1.1 GB**  
- Chunked prefill (512) makes long context interactive on WDDM  

**Next (goal-aligned):** streaming compress (cut peak VRAM); 3-depth L_ε; multi-hop (H3).

---

## Docs

| File | Content |
|------|---------|
| `RESEARCH_BRIEF.md` | Literature notes, early invention framing |
| `papers/NOTES.md` | Paper cards (SnapKV, PyramidKV, Ada-KV, KIVI, RocketKV, …) |
| `results/FINDINGS.md` | Experimental findings |

Related work is **crowded** (eviction + KV quant). We treat known methods as **baselines/tools**. Success is **raising \(L\) at ε≈0**, not renaming SnapKV.

---

## Repo

Private: [nearlossless-context](https://github.com/nilsperssonsuorra/nearlossless-context)

```text
nearlossless-context/
  experiments/     # benches + methods
  papers/          # notes
  results/         # local CSVs + FINDINGS.md
  RESEARCH_BRIEF.md
  requirements.txt
```
