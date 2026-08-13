"""Runnable end-to-end example for the public package API."""

from __future__ import annotations

import argparse
from typing import Sequence


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small long-context retrieval example with streaming KV-cache "
            "compression. The first run downloads the selected Hugging Face model."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Approximate prompt length (default: 768 on CPU, 1536 on CUDA)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum number of greedily generated tokens",
    )
    return parser


def make_document(repetitions: int) -> str:
    """Build repetitive filler with a retrieval-critical fact near the middle."""
    filler = (
        "The archive contains routine operational notes about weekly maintenance, "
        "inventory checks, and ordinary status reports. "
    )
    before = filler * max(1, repetitions // 2)
    after = filler * max(1, repetitions - repetitions // 2)
    fact = "The unique recovery code for Project Alder is ALDER-7391. "
    return before + fact + after


def build_input_ids(tokenizer, target_tokens: int):
    """Create a chat-formatted prompt at or above the requested length."""
    if target_tokens < 576:
        raise ValueError("--prompt-tokens must be at least 576 to demonstrate compression")

    repetitions = 8
    while True:
        content = (
            "Read the archive below and answer the question using only its contents.\n\n"
            f"{make_document(repetitions)}\n\n"
            "Question: What is the recovery code for Project Alder? "
            "Answer with only the code."
        )
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if int(input_ids.shape[-1]) >= target_tokens:
            return input_ids
        repetitions *= 2


def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import greedy_generate, prefill_auto

    device = _resolve_device(args.device, torch.cuda.is_available())
    target_tokens = args.prompt_tokens or (1536 if device == "cuda" else 768)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    version_parts = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    dtype_arg = "dtype" if version_parts >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_arg: dtype})
    model.to(device)
    model.eval()

    input_ids = build_input_ids(tokenizer, target_tokens).to(device)
    prompt_tokens = int(input_ids.shape[-1])

    past, logits, info = prefill_auto(
        model,
        input_ids,
        mode="stream",
        tokenizer=tokenizer,
        discovery="novelty",
        chunk_size=256,
    )
    generated = greedy_generate(
        model,
        past,
        logits,
        args.max_new_tokens,
        eos_id=tokenizer.eos_token_id,
        next_position=prompt_tokens,
    )

    stats = info["stats"]
    policy = info["policy"]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    retained = int(stats["final_cache"])

    print("\nCompression summary")
    print(f"  path:            {info['path']}")
    print(f"  model family:    {policy['family']}")
    print(f"  original tokens: {prompt_tokens}")
    print(f"  retained tokens: {retained}")
    print(f"  peak cache:      {stats['peak_cache']}")
    print(f"  stream budget:   {policy['stream_budget']}")
    print(f"  compressions:    {stats['n_compress']}")
    print(f"  retained ratio:  {retained / prompt_tokens:.1%}")
    print(f"\nAnswer: {answer}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
