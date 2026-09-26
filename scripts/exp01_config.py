"""Fixed parameters for exp01_organism_gap, shared by every exp01 script.

Standard library only: the vLLM venv and /venv/main both import this module.
Everything that defines the experiment lives here, so the manifest can be
built from it and no script carries its own copy of a seed or a path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXP_ID = "exp01_organism_gap"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / EXP_ID
CACHE_DIR = REPO_ROOT / "cache" / "exp01"
LOG_DIR = REPO_ROOT / "log" / "exp01"
FIGURES_DIR = REPO_ROOT / "figures"
MODELS_ROOT = Path("/workspace/models")
VLLM_PYTHON = REPO_ROOT / "scripts" / "vllm_python.sh"

# Intermediate files (cache/, gitignored).
PROMPTS_PATH = CACHE_DIR / "prompts.jsonl"
PROMPT_PREP_PATH = CACHE_DIR / "prompt_prep.json"
DATASET_CACHE_DIR = CACHE_DIR / "dataset"

# Result files (results/, committed).
MANIFEST_PATH = RESULTS_DIR / "manifest.json"
LOAD_REPORT_PATH = RESULTS_DIR / "load_report.json"
GATE_REPORT_PATH = RESULTS_DIR / "gate_report.json"
GENERATIONS_PATH = RESULTS_DIR / "generations.jsonl"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
READ_SAMPLES_PATH = RESULTS_DIR / "read_samples.md"
FIGURE_PATH = FIGURES_DIR / f"{EXP_ID}_type_hint_rates.png"

# --- Models --------------------------------------------------------------
# Checkpoint IDs in the reference repo's manifests/checkpoints/checkpoints_v1.yaml.
BASE_CHECKPOINT_ID = "ckpt_qwen35_4b_base_v1"
ORGANISM_CHECKPOINT_ID = "ckpt_qwen35_hua_stage6b_adapter_stack_v1"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_REPO = "Luxel/qwen35-hua-stage6b-organism-adapter-stack-v1"
ADAPTER_REVISION = "bb03a17b2cf78fde27b8059130fc5c3ed0bfd3f6"
ADAPTER_ORDER = [
    "01_sdf_ei_tinker_peft_hf_current",
    "02_stage3_dpo_lora_soft_v1",
    "03_stage6_ei_replay_dpo_ckpt20",
    "04_stage6b_ei_replay_dpo_ckpt20",
]
# Adapter 01 was trained with separate in_proj_q/k/v on the linear-attention
# layers; Qwen3.5 in transformers fuses them into in_proj_qkv, so those LoRA
# tensors match no module and are dropped (24 layers x 3 projections x A/B).
EXPECTED_DROPPED_LORA_TENSORS = {
    "01_sdf_ei_tinker_peft_hf_current": 144,
    "02_stage3_dpo_lora_soft_v1": 0,
    "03_stage6_ei_replay_dpo_ckpt20": 0,
    "04_stage6b_ei_replay_dpo_ckpt20": 0,
}
MODEL_VARIANTS = {
    "organism": MODELS_ROOT / "organism-merged",
    "base": MODELS_ROOT / "base-text",
}

# --- Data and conditions -------------------------------------------------
DATASET_REPO = "ObscurumLight/eval-aware-qwen3.5-4b"
DATASET_REVISION = "ba727e2782d10b54390424674e85724252414c9e"
DATASET_SUBDIR = "datasets/ds_eval_awareness_hua_repro_v2"
DATASET_FILES = ["samples.csv", "splits.csv", "summary.json"]
DATASET_ID = "ds_eval_awareness_hua_repro_v2"
SPLIT_ID = "split_eval_awareness_hua_repro_v2_all"
CONDITION_SET_ID = "baseline_hua_main_with_sa_hint_v2"
CONDITIONS = {"EVAL": "evaluation_evaluation", "DEPLOY": "deployment_deployment"}
THINKING_MODE = "qwen_thinking_on"

HELDOUT_N_TASKS = 100
PILOT_N_TASKS = 50
TASK_SAMPLE_SEED = 20260925

# --- Decoding arms -------------------------------------------------------
# Arm B is the headline. Arm A reuses the card's decoding for comparison.
ARMS = {
    "A_card_decoding": {
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "B_qwen_official": {
        "n": 2,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 32768,
    },
}
HEADLINE_ARM = "B_qwen_official"
VLLM_ENGINE_SEED = 42
VLLM_MAX_MODEL_LEN = 36864


def prompt_seed(sample_id: str) -> int:
    """Per-prompt sampling seed, identical for both model variants."""
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)


def raw_generations_path(variant: str) -> Path:
    """Content-addressed cache file for one variant's raw vLLM outputs.

    The key covers the saved weights, the exact prompt tokens and the decoding
    arms, so any change to those lands in a new file instead of overwriting.
    """
    load_report = json.loads(LOAD_REPORT_PATH.read_text())
    prompt_prep = json.loads(PROMPT_PREP_PATH.read_text())
    key_material = {
        "weights_sha256": load_report["saved"][variant]["file_sha256"],
        "prompts_sha256": prompt_prep["prompts_file_sha256"],
        "arms": ARMS,
        "engine_seed": VLLM_ENGINE_SEED,
    }
    key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return CACHE_DIR / f"raw_generations_{variant}_{key}.jsonl"


# --- Token IDs (Qwen3.5 tokenizer; verified in exp01_prepare_prompts.py) --
IM_END_TOKEN_ID = 248046
THINK_START_TOKEN_ID = 248068
THINK_END_TOKEN_ID = 248069

# --- Logprob gate (pre-registered before any generation) ------------------
GATE = {
    "n_prompts": 4,
    "n_continuation_tokens": 128,
    "prompt_selection_seed": 20260925,
    "relative_factor": 2.0,
    "absolute_floor_nats": 0.05,
    "rule": (
        "pass iff organism mean |logprob_hf - logprob_vllm| <= "
        "max(relative_factor * base mean |logprob_hf - logprob_vllm|, absolute_floor_nats)"
    ),
}
PREFILL_DIAGNOSTIC_TOKENS = 8

# --- Analysis ------------------------------------------------------------
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 20260506
CI_LEVEL = 0.95
READ_SAMPLE_SEED = 20260925
READ_SAMPLES_PER_CONDITION = 5

CARD_REFERENCE = {
    "source": "organism_stack.json in the adapter repo (evaluate_qwen35_stage3_dpo.py)",
    "n_tasks": 100,
    "decoding": "tinker_cookbook qwen3 renderer, greedy, max_new_tokens 512, seed 42",
    "scorer": "regex has_type_hint on the renderer-parsed response",
    "rates": {"DEPLOY": 0.050, "EVAL": 0.340},
    "gap_eval_minus_deploy": 0.290,
}
