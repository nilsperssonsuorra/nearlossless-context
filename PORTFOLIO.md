# Portfolio note — nearlossless-context

**One-liner:** Near-lossless long context on a 24 GB GPU by keeping **critical local neighborhoods**, diagnosing stream failures as a **query-unknown discovery** problem, closing most of that gap with **sticky surface novelty** (multi-seed **40k** at peak ~1k), and showing on public LongBench that **posthoc ≈ full** while online quality is a **peak/quality Pareto** (~0.92× full F1 at ~2.5k peak).

---

## Ready-to-paste blurb (LinkedIn / email / intro)

I ran a private research lab on near-lossless long-context inference under fixed VRAM (RTX 3090 24 GB, Qwen3-4B). Multi-seed kill tests showed that near–full-KV retrieval needs critical fact tokens plus a small local radius—not bare spans and not the whole haystack (15/15 oracle, 0/15 anti-oracle). Online streaming failed under hay variation at moderate budgets not because peak cache was too small, but because query-unknown discovery is weak: attention-based stream@512 was only ~33% multi-seed while perfect online pin of the critical neighborhood was 15/15 at the same peak. I built a training-free surface-novelty detector with sticky pin packing that closes most of that gap—multi-seed success through 40k tokens at ~1k peak cache, multi-hop and multi-document stresses, and an external-style mixed QA slice where novelty matched full-context hit rate (10/10) while attention streaming scored 0/10. Peak decode KV stayed ~72 MB and peak VRAM ~8.3 GB flat from 4k to 40k. Writeup, figure, and limitations are in the repo paper draft; this is a workstation mechanism/systems study, not a general long-context leaderboard claim.

*(~130 words — paste as-is or trim the last sentence for shorter posts.)*

### Ultra-short (2 sentences)

I studied near-lossless KV compression on a 24 GB GPU and showed that stream failures are mostly a **query-unknown discovery** problem, not just budget size. A sticky surface-novelty detector recovers multi-seed quality through **40k** context at ~**1k** peak cache tokens (~72 MB decode KV, ~8.3 GB peak VRAM flat), matching full quality on a mixed external-style QA slice where attention streaming fails.

### CV bullet form

- Built multi-seed near-lossless long-context lab (Qwen3-4B, RTX 3090): proved critical±radius necessity/sufficiency (15/15 vs 0/15 anti-oracle).
- Identified stream bottleneck as query-unknown discovery (oracle pin 15/15 vs attn stream 33% @512).
- Implemented sticky surface-novelty KV streaming: multi-seed ε≈0 through 40k @ ~1k peak cache; multi-hop/multi-doc stresses; external-style QA 10/10 hits (= full) vs valley 0/10.
- Systems: decode KV ~72 MB, peak VRAM ~8.3 GB flat 4k→40k; hybrid Gemma-4 transfer; adaptive `prefill_auto` API + paper draft with limitations.

---

## Artifacts to open first

| Artifact | Why |
|----------|-----|
| [`papers/main.pdf`](papers/main.pdf) | Readable draft + limitations |
| [`papers/figures/fig1_story.png`](papers/figures/fig1_story.png) | H1 kill → discovery gap → long-L / peak cache |
| [`results/FINDINGS.md`](results/FINDINGS.md) | Full measured tables |
| [`README.md`](README.md) | Headline results + how to run |

## Claims (supported)

1. **Structure:** critical tokens + small radius \(R^*=1\) are necessary and sufficient for ε≈0 single-fact recall (multi-seed 15/15 / 0/15 kill).
2. **Discovery gap:** perfect online pin hits 15/15 at stream@512 where attention valley only gets ~33%.
3. **Sticky novelty:** query-unknown surface detector restores multi-seed quality through **40k** at flat peak cache ~1k; multi3/hop2/hop3/prose/multidoc stress strong vs valley.
4. **External-style / public slice:** offline mixed QA — novelty **10/10** (= full) vs valley **0/10**. **Public LongBench** (60 items @4k truncate): full mean F1 **0.28** / hit **25%**; novelty **~0.18** / 10% — ≪ full online; **posthoc query-aware ~0.95× full** @ final 512; **query_hold best ~0.92×** @ peak ~2.5k.
5. **Systems:** decode KV ~72 MB and peak VRAM ~8.3 GB stay flat 4k→40k under novelty stream@512.

## Do not overclaim

Not general long-context SOTA. Public LongBench slice shows novelty helps only slightly vs attention streaming and lags full context — method is strong on retrieval/needle-class tasks, weaker on open long-doc QA. Not production kernels. See paper §Limitations.

## Code entry

```python
from compress_adaptive import prefill_auto
past, logits, info = prefill_auto(model, input_ids, mode="stream", discovery="novelty")
```

Lab: RTX 3090 24 GB · primary `Qwen/Qwen3-4B-Instruct-2507` · private repo `nearlossless-context`.
