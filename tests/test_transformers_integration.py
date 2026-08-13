from __future__ import annotations

import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from nearlossless_context import greedy_generate, prefill_auto
from nearlossless_context.snapkv import cache_seq_len


class _SurfaceTokenizer:
    """Minimal tokenizer surface used by novelty scoring; no model files needed."""

    eos_token_id = None

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        token_id = int(token_ids[0])
        if token_id in {101, 102, 103}:
            return f"ALDER-{token_id}"
        return " routine"


def _tiny_qwen2() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_real_dynamic_cache_compresses_and_decodes_offline() -> None:
    model = _tiny_qwen2()
    tokenizer = _SurfaceTokenizer()
    input_ids = torch.full((1, 576), 7, dtype=torch.long)
    input_ids[0, 260:263] = torch.tensor([101, 102, 103])
    input_ids[0, -8:] = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18])

    past, logits, info = prefill_auto(
        model,
        input_ids,
        mode="stream",
        tokenizer=tokenizer,
        discovery="novelty",
        chunk_size=256,
        model_id="Qwen/Qwen2.5-tiny-integration-test",
    )

    budget = int(info["policy"]["stream_budget"])
    stats = info["stats"]

    assert {"path", "L", "policy", "model_id"} <= info.keys()
    assert {
        "discovery",
        "n_entities_hat",
        "stats",
        "logical_kv_mb_int8",
        "use_int8",
    } <= info.keys()
    assert {"stream_budget", "final_budget", "n_compress"} <= stats.keys()
    assert isinstance(past, DynamicCache)
    assert int(stats["n_compress"]) >= 1
    assert int(stats["peak_cache"]) > budget
    assert int(stats["final_cache"]) <= budget
    assert cache_seq_len(past) == int(stats["final_cache"])

    generated = greedy_generate(
        model,
        past,
        logits,
        2,
        eos_id=None,
        next_position=int(input_ids.shape[-1]),
    )

    assert len(generated) == 2
    assert all(isinstance(token_id, int) for token_id in generated)


def test_full_and_posthoc_metadata_contracts_offline() -> None:
    input_ids = torch.arange(64, dtype=torch.long).unsqueeze(0) % 128
    common = {"path", "L", "policy", "model_id"}

    _, _, full_info = prefill_auto(
        _tiny_qwen2(),
        input_ids,
        mode="full",
        chunk_size=32,
        model_id="Qwen/Qwen2.5-tiny-integration-test",
    )
    assert common <= full_info.keys()
    assert {"cache_tokens"} <= full_info.keys()
    assert full_info["policy"] is None
    assert full_info["cache_tokens"] == 64

    _, _, posthoc_info = prefill_auto(
        _tiny_qwen2(),
        input_ids,
        mode="posthoc",
        chunk_size=32,
        window_size=16,
        n_entities=1,
        model_id="Qwen/Qwen2.5-tiny-integration-test",
    )
    assert common <= posthoc_info.keys()
    assert {
        "n_entities_hat",
        "keep_count",
        "cache_tokens",
        "logical_kv_mb_int8",
        "use_int8",
    } <= posthoc_info.keys()
    assert posthoc_info["policy"] is not None
