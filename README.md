# ByteBudgetKV — cheap long context (lab)

**Hardware:** RTX 3090 24 GB + 32 GB RAM  
**Phase 1 model:** `Qwen/Qwen3-4B-Instruct-2507` only (pure full-attention GQA)  
**Note:** `Qwen3.5-4B` is hybrid linear+full attention — awkward for classic KV eviction; use later if we extend the method.  
**Goal:** Prove that under a fixed **byte** KV budget, hetero (tokens × precision) per head/layer beats uniform baselines — quality + flatter decode as context grows.

See `RESEARCH_BRIEF.md` for literature and invention claim.

## Setup

```powershell
cd "X:\Upscaler\Egna\0. Random games\researchcontext"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
# Install a CUDA build of torch matching your driver if pip's default is wrong:
# https://pytorch.org/get-started/locally/
```

## Phase 1 — measure the long-context tax

```powershell
python experiments\bench_context_tax.py
# shorter smoke:
python experiments\bench_context_tax.py --ctx 2048,4096,8192
```

## Phase 1b — full KV vs SnapKV vs recent-window

**Stay ≤4k context** on this workstation (8k+ full KV thrash / lag under WDDM).

```powershell
python experiments\bench_compare.py --ctx 2048,4096 --budget 1024
```

Outputs land in `results/*.csv` (+ `.json` meta).

## Needle quality smoke (≤4k)

```powershell
python experiments\bench_needle.py --ctx 4096 --depths 0.0,0.5,1.0 --budget 1024 --window 128
```

## Equal-budget Pareto

```powershell
python experiments\bench_equal_byte.py --ctx 4096 --depths 0.0,0.5,1.0 --budgets 512,1024,1536
```

Latest: SnapKV **9/9** needles; ByteBudget **7/9** but much smaller cache @1024. See `results/FINDINGS.md`.
