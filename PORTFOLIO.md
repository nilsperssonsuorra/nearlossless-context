# Portfolio note — nearlossless-context

**One-liner:** Near-lossless long context on a 24 GB GPU by keeping **critical local neighborhoods**, diagnosing stream failures as a **query-unknown discovery** problem, and closing most of that gap with **sticky surface novelty** (through multi-seed **40k** at peak cache ~1k tokens).

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
4. **External-style slice:** 10 mixed-domain QA items padded to ~4k — novelty **10/10** substring hits (= full); valley **0/10** at stream@512.
5. **Systems:** decode KV ~72 MB and peak VRAM ~8.3 GB stay flat 4k→40k under novelty stream@512.

## Do not overclaim

Not general long-context SOTA; not full RULER/LongBench leaderboard; not production kernels. Suite is retrieval-heavy with honest multi-seed \(N\). See paper §Limitations.

## Code entry

```python
from compress_adaptive import prefill_auto
past, logits, info = prefill_auto(model, input_ids, mode="stream", discovery="novelty")
```

Lab: RTX 3090 24 GB · primary `Qwen/Qwen3-4B-Instruct-2507` · private repo `nearlossless-context`.
