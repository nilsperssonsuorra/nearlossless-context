# Paper result provenance

`paper_aggregates.csv` is the machine-readable source for the principal numbers in
`papers/main.tex` and Figure 1. Each row names the immutable, timestamped local run
summary from which it was copied. The raw result files are excluded from Git because
they can be large and may contain benchmark examples or model generations.

The central protocols are:

- H1 and scorer tax: `paper_rigor_20260716T120146Z.json`, Qwen3-4B, context 4096,
  seeds 0--4, depths 0, 0.5, and 1.
- Discovery: `capsules_20260716T164126Z.json`, the same 15 cells.
- Long-context streaming: `novelty_longL_*.json`, seeds 0--2 and the same three
  depths. The attention-valley baseline was measured through 16k; only sticky
  novelty was measured at 24k, 32k, and 40k.
- Public QA: `external_slice_20260717T222040Z.json` for posthoc compression and
  `external_slice_20260717T225301Z.json` for online novelty/query-hold. Both use
  the same fixed 60-item LongBench slice.
- Resource measurements: `systems_resources_20260717T130143Z.json`; each length
  is one seed-0, mid-depth run, so these are operating-point measurements rather
  than replicated latency estimates.

Regenerate the figure with `python experiments/plot_paper_figures.py`. The script
reads `paper_aggregates.csv` instead of embedding the plotted measurements.
