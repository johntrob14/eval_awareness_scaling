"""Variants (base model x precision condition) and paths for the quantization study.

A variant is named "<base>@<condition>", e.g. "Qwen3.5-4B@gptq-w4". Bases are the ladder's subjects
(ladder/scripts/ladder_config.py), so prompts, sampling, caps and per-rollout seeds are the ladder's own.

Paths: QUANT_DATA (default data/quant) holds rollouts/ and judgments/ in the unified format; QUANT_CHECKPOINTS
(default checkpoints/) holds our quantized weights. On Lonestar6 both live under $WORK/results (.sauce/env.sh).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LADDER_SCRIPTS = ROOT / "ladder" / "scripts"
DATA = Path(os.environ.get("QUANT_DATA", ROOT / "data" / "quant"))
CHECKPOINTS = Path(os.environ.get("QUANT_CHECKPOINTS", ROOT / "checkpoints"))
SPLITS = {"adversarial": ("fortress_adv", 10), "benign": ("fortress_benign", 2)}  # prompt set, rollouts per item

# Calibration data for our checkpoints: RedHat's Qwen3.5 recipe (raw instruction text, no chat template).
CALIBRATION = {"repo": "garage-bAInd/Open-Platypus", "revision": "37141edbdb7826378cce118c46a109b813e1f038",
               "n_samples": 1024, "max_seq_len": 8192}

# Our recipes. "calibration_slice" picks rows of Open-Platypus train; slice B is the noise-floor replicate.
RECIPES = {
    "gptq-w8": {"method": "gptq", "scheme": "W8A16", "calibration_slice": (0, 1024)},
    "gptq-w4": {"method": "gptq", "scheme": "W4A16", "calibration_slice": (0, 1024)},
    "gptq-w4-calB": {"method": "gptq", "scheme": "W4A16", "calibration_slice": (1024, 2048)},
    "rtn-w4": {"method": "rtn", "scheme": "W4A16", "calibration_slice": None},
}
GPTQ_DAMPENING = 0.05  # RedHat's value for Qwen3.5
# Kept in bf16. Qwen3.5: the Gated-DeltaNet linear attention, the vision tower and MTP heads (as RedHat does).
IGNORE = ["re:.*lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*linear_attn.*", "re:.*mtp.*"]

# Off-the-shelf 4-bit checkpoints (what people download), pinned 2026-11-05.
OFF_THE_SHELF = {
    "Qwen3.5-4B": ("RedHatAI/Qwen3.5-4B-quantized.w4a16", "7a613872f394578b0b52b683ff4ac47516b4bcaf"),
    "Qwen3.5-9B": ("RedHatAI/Qwen3.5-9B-quantized.w4a16", "a398088c4228b0ae0c8c78df88fd1e4bf445f068"),
    "Qwen3.5-27B": ("Qwen/Qwen3.5-27B-GPTQ-Int4", "8f0c09f227ae570e79617c6d9172b59df9c16081"),
    "Qwen3-8B": ("RedHatAI/Qwen3-8B-quantized.w4a16", "32527053243a382bc56e941c964fc516a5014d39"),
    "Qwen3-14B": ("RedHatAI/Qwen3-14B-quantized.w4a16", "98ce4ba52bc02e37e2beb0b0a5f11d7f0e589685"),
    "Qwen3-32B": ("RedHatAI/Qwen3-32B-quantized.w4a16", "0364e6546d36447655a78bac8ab181572fd6ff24"),
}
CONDITIONS = ["bf16", *RECIPES, "hf-w4"]


def ladder_config():
    if str(LADDER_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(LADDER_SCRIPTS))
    import ladder_config
    return ladder_config


def split_variant(variant: str) -> tuple[str, str]:
    base, _, condition = variant.partition("@")
    cfg = ladder_config()
    if base not in cfg.ALL_SUBJECTS:
        raise SystemExit(f"unknown base model {base!r}; choose from {list(cfg.ALL_SUBJECTS)}")
    if condition not in CONDITIONS:
        raise SystemExit(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    if condition == "hf-w4" and base not in OFF_THE_SHELF:
        raise SystemExit(f"no off-the-shelf checkpoint listed for {base}")
    return base, condition


def checkpoint_dir(base: str, condition: str) -> Path:
    return CHECKPOINTS / f"{base}@{condition}"


def weights(variant: str) -> dict:
    """Where a variant's weights are, and what they are (recorded in every run manifest)."""
    base, condition = split_variant(variant)
    cfg = ladder_config()
    model = cfg.ALL_SUBJECTS[base]
    bf16 = cfg.model_dir(model["repo"], model["revision"])
    if condition == "bf16":
        return {"path": bf16, "repo": model["repo"], "revision": model["revision"], "tokenizer": bf16}
    if condition == "hf-w4":
        repo, revision = OFF_THE_SHELF[base]
        return {"path": cfg.model_dir(repo, revision), "repo": repo, "revision": revision, "tokenizer": bf16}
    return {"path": checkpoint_dir(base, condition), "recipe": condition, "tokenizer": bf16}


def rollouts_path(variant: str, split: str) -> Path:
    return DATA / "rollouts" / f"{variant}.{split}.jsonl"
