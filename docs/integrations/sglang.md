# SGLang KV-sparsity integration assessment

Status: feasibility assessment for [issue #4](https://github.com/nilsperssonsuorra/nearlossless-context/issues/4), based on SGLang `main` and its open [unified KV-sparsity RFC](https://github.com/sgl-project/sglang/issues/32657) as inspected on 2026-08-20.

## Conclusion

The surface-novelty policy can plausibly participate in SGLang's proposed
logical-selection layer, but the current integration boundary is not sufficient
for the full fixed-VRAM method.

A first integration should be **visibility-only**: produce a bounded,
fixed-capacity set of logical token or page indices and let an SGLang backend
restrict which resident KV entries attention can see. This can validate policy
semantics and serving-path correctness. It does **not** reclaim KV memory and
must not be presented as increasing context capacity or request concurrency.

The paper's main systems claim depends on physical retention: unselected KV is
removed or compacted so storage can be reused. SGLang's RFC explicitly leaves
physical eviction, allocator updates, and reclaimed HBM to later work.

## Policy requirements

The current implementation is centered on
[`score_novelty`](../../experiments/novelty_detect.py),
[`novelty_abs_set`](../../experiments/novelty_detect.py), and
[`prefill_streaming_novelty_pin`](../../experiments/novelty_detect.py). It needs:

- request token IDs, including the prefix processed before the current forward;
- tokenizer decoding or an equivalent token-ID-to-surface-feature table;
- prefix token frequencies and first-occurrence positions;
- absolute logical positions;
- a per-request sticky-pin set that survives chunked prefill;
- sink and recent-window positions reserved before novelty pins;
- deterministic packing into a fixed budget.

The novelty signal is query-unknown and does not need attention matrices. It is
also model-family specific because token surface pieces come from the model's
tokenizer.

## Mapping to the proposed SGLang model

| Dimension | Surface-novelty policy | Compatibility |
|---|---|---|
| KV-state action | Physical eviction/compaction in the measured method | Outside the RFC's initial visibility-only scope |
| Selection granularity | Logical token positions | Direct only for token-granular backends; page backends require explicit page expansion |
| Selection evidence | Token IDs, tokenizer-derived surface cues, and prefix frequency state | Query-independent, but token-derived evidence is not yet an explicit policy input contract |
| Decision scope | Updated after each prefill chunk; sticky state is reused for the request | Fits request state plus incremental prefill lifecycle hooks |
| Output shape | Bounded selected indices plus a valid length | Fits the RFC's proposed fixed-capacity selection result |
| Model scope | Dense MHA/GQA and tested hybrid full-attention layers | Broadly aligned, subject to tokenizer and backend support |

### Token availability

SGLang's current `ForwardBatch` includes `input_ids`, request-pool indices, and
sequence lengths. During chunked prefill, those input IDs describe the tokens in
the current forward, not a stable public contract for reconstructing the full
request prefix inside a sparsity policy.

The current `SparseCoordinator.on_request_begin` hook receives a request whose
`origin_input_ids` are available, but the coordinator currently records only the
prompt length. A novelty policy therefore needs one of these explicit designs:

1. copy request token IDs into generation-aware policy state at request start and
   append generated or newly extended IDs at later forwards; or
2. receive a read-only request token-history view from the serving runtime.

Option 2 is cleaner for a shared upstream framework. Either option must define
cleanup, request-slot reuse, multimodal/embedding-only requests, and tokenizer
ownership. An embedding-only request cannot use surface novelty and should fall
back to dense attention or another supported policy.

### Token versus page selection

The current SGLang `FlashAttentionAdaptor` translates logical pages to physical
pages. Surface novelty returns token positions. Exact conversion is possible
with page size 1. For larger pages, selecting every page containing at least one
chosen token preserves the chosen tokens but over-selects their neighbors.

That conversion must be declared and measured; it must not be hidden as though
token- and page-level budgets were equivalent. A first prototype should either:

- require page size 1 for exact semantic tests; or
- expose both the token budget and resulting page-expanded KV budget.

### Request identity and state

Sticky pins, token counts, and first-occurrence positions are request state.
They must be keyed by a generation-aware request identity, not only a reusable
request-pool slot. Cleanup on completion and abort is required to prevent one
request from inheriting another request's novelty state.

## Smallest honest end-to-end path

The first useful prototype should deliberately exclude physical eviction:

1. Add a surface-novelty policy that incrementally maintains token counts,
   surface features, and sticky logical positions per request.
2. Pack sinks, recent tokens, and ranked novelty positions into a deterministic
   fixed-capacity tensor with a device-resident valid length.
3. Adapt token positions to one supported backend, initially with page size 1
   or explicitly reported page expansion.
4. Run a dense-versus-sparse correctness smoke test on one dense MHA/GQA model.
5. Report retained logical tokens, visible KV tokens/pages, latency, and answer
   correctness. Do not report reclaimed HBM or increased concurrency.
6. Preserve dense fallback for unsupported requests and configurations.

Only after that path works should a separate design address physical KV
placement, allocator ownership, prefix-cache interaction, batching, and CUDA
Graph-safe memory reuse.

## Prototype decision

A tested adapter in this repository is premature because SGLang's public RFC
contract is still being designed and the current source interface does not
provide a stable full-prefix token-history abstraction to policies. Implementing
an independent look-alike `SelectionResult` here would test our invented API,
not the upstream serving contract.

The next code milestone should begin only after upstream confirms the intended
token-history and token-to-page contracts. Until then, this assessment records
why the adapter portion of issue #4 is blocked and what would unblock it.

## Draft upstream question

The following is suitable for a focused comment on the SGLang RFC after a final
review against its latest discussion:

> I maintain `nearlossless-context`, which includes a query-unknown
> surface-novelty retention policy for dense MHA/GQA models. The selector incrementally
> uses request token IDs, tokenizer-derived surface features, prefix frequency
> state, and sticky logical positions; it does not require attention matrices.
>
> Would token-ID-derived evidence be in scope for the initial `SparsityPolicy`
> contract? In particular, would the framework prefer a generation-aware,
> read-only request token-history view, rather than each policy copying token
> IDs from request/forward hooks?
>
> The policy naturally returns fixed-capacity logical token indices. For a
> page-only adaptor, should capability negotiation require page size 1 for exact
> mapping, or allow explicit page expansion with both token and page budgets
> reported? I can prototype a visibility-only path and correctness test without
> making HBM-reclamation claims; physical eviction would remain separate.

## Upstream references

- [Unified KV-cache sparsity RFC](https://github.com/sgl-project/sglang/issues/32657)
- [`BaseSparseAlgorithm`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py)
- [`SparseCoordinator`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py)
- [`BackendAdaptor`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/sparsity/backend/backend_adaptor.py)
- [`ForwardBatch`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py)
