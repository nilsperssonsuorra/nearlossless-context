"""Project defaults — single primary model for theory work."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# PRIMARY: pure full-attention GQA (good for classic KV research).
# NOTE: Qwen/Qwen3.5-4B is hybrid linear+full attention — bad first target for SnapKV/ByteBudgetKV.
PRIMARY_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# TRANSFER: second full-attention model (not hybrid) for cross-model smoke.
TRANSFER_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# Out-of-family transfer target (full attn).
OUT_OF_FAMILY_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Context lengths for tax / compare curves
# Cap at 4k on this workstation — 8k+ pegs the 3090 and freezes the desktop (WDDM).
DEFAULT_CTX_LENGTHS = [2048, 4096]

# Generation tokens for decode speed measurement
DECODE_NEW_TOKENS = 64

# SnapKV-style defaults
# Observation window must cover the user question (not just a few tokens).
SNAPKV_WINDOW = 128
SNAPKV_MAX_CAPACITY = 1024  # total KV tokens kept (obs window included)
SNAPKV_KERNEL = 7

FILLER_UNIT = "The quick brown fox jumps over the lazy dog. "
