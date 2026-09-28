"""vLLM engine construction shared by the exp01 gate and generation scripts (vLLM venv only)."""

from __future__ import annotations

from pathlib import Path

import exp01_config as cfg


def make_llm(model_dir: Path):
    from vllm import LLM

    return LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="bfloat16",
        seed=cfg.VLLM_ENGINE_SEED,
        max_model_len=cfg.VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        # Sampling parameters are always passed explicitly; do not inherit model defaults.
        generation_config="vllm",
        enable_prefix_caching=False,
    )
