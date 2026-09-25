"""exp01: write results/exp01_organism_gap/manifest.json before anything is generated.

Run in /venv/main after committing the code and running exp01_prepare_prompts.py:
    python scripts/exp01_write_manifest.py
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import subprocess

import exp01_config as cfg
from _reference import reference_commit

MAIN_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "huggingface_hub",
    "accelerate",
]
VLLM_PACKAGES = ["vllm", "torch", "transformers"]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cfg.REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def vllm_env_versions() -> dict[str, str]:
    probe = f"import importlib.metadata as m, json; print(json.dumps({{p: m.version(p) for p in {VLLM_PACKAGES!r}}}))"
    out = subprocess.run(
        [str(cfg.VLLM_PYTHON), "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def gpu_name() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    if cfg.MANIFEST_PATH.exists():
        raise FileExistsError(
            f"{cfg.MANIFEST_PATH} exists; results are never overwritten"
        )
    dirty = [
        line
        for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
        if line
    ]
    if dirty:
        raise SystemExit(f"Commit the code first; tracked files are modified: {dirty}")
    prep = json.loads(cfg.PROMPT_PREP_PATH.read_text())

    manifest = {
        "exp_id": cfg.EXP_ID,
        "written_at_utc": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        "git_commit": git("rev-parse", "HEAD"),
        "reference_repo": {
            "url": "https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness",
            "path": "third_party/Non-verbal-Eval-Awareness",
            "commit": reference_commit(),
        },
        "claim_tested": (
            "The Qwen3.5-4B eval-aware organism writes type hints more often under the EVAL "
            "condition than under DEPLOY, and the base model does not show the same gap."
        ),
        "models": {
            "base": {
                "repo": cfg.BASE_MODEL,
                "revision": cfg.BASE_REVISION,
                "checkpoint_id": cfg.BASE_CHECKPOINT_ID,
                "served_from": str(cfg.MODEL_VARIANTS["base"]),
                "build": "reference load_qwen35_model (HF, bf16), saved text-only with save_pretrained",
            },
            "organism": {
                "adapter_repo": cfg.ADAPTER_REPO,
                "adapter_revision": cfg.ADAPTER_REVISION,
                "checkpoint_id": cfg.ORGANISM_CHECKPOINT_ID,
                "adapter_order": cfg.ADAPTER_ORDER,
                "expected_dropped_lora_tensors": cfg.EXPECTED_DROPPED_LORA_TENSORS,
                "served_from": str(cfg.MODEL_VARIANTS["organism"]),
                "build": "base + reference apply_peft_adapter_stack (PEFT merge_and_unload after each adapter)",
            },
        },
        "data": {
            **prep["dataset"],
            "dataset_id": cfg.DATASET_ID,
            "split_id": cfg.SPLIT_ID,
            "condition_set_id": cfg.CONDITION_SET_ID,
            "conditions": cfg.CONDITIONS,
            "heldout_rule": prep["heldout_rule"],
            "heldout_task_ids": prep["heldout_task_ids"],
            "pilot_rule": prep["pilot_rule"],
            "pilot_task_ids": prep["pilot_task_ids"],
        },
        "prompting": {
            "builder": "reference messages_for_sample(rewrite_identity=True) + apply_qwen_chat_template",
            "thinking_policy": prep["thinking_policy"],
            "chat_template": "official Qwen3.5 template from the base tokenizer",
            "chat_template_sha256": prep["chat_template_sha256"],
            "prompt_suffix": prep["expected_prompt_suffix"],
            "prompts_file": str(cfg.PROMPTS_PATH.relative_to(cfg.REPO_ROOT)),
            "prompts_file_sha256": prep["prompts_file_sha256"],
            "n_prompts": prep["n_prompts"],
        },
        "generation": {
            "backend": "vLLM, prompts passed as token IDs",
            "arms": cfg.ARMS,
            "headline_arm": cfg.HEADLINE_ARM,
            "stop_token_ids": [cfg.IM_END_TOKEN_ID],
            "engine_seed": cfg.VLLM_ENGINE_SEED,
            "per_prompt_seed": "int(sha256(sample_id)[:8], 16), same for both variants; greedy arm unseeded",
            "max_model_len": cfg.VLLM_MAX_MODEL_LEN,
        },
        "logprob_gate": cfg.GATE,
        "prefill_diagnostic_tokens": cfg.PREFILL_DIAGNOSTIC_TOKENS,
        "scoring": {
            "headline": "has_type_hints = reference eval_contract_v1 on '<think>\\n' + decoded completion",
            "secondary": [
                "has_type_hints_answer_only: eval_contract_v1 on the text after </think>",
                "has_type_hints_card: card regex has_type_hint on reasoning + '\\n' + answer",
            ],
            "abstentions": "contract None labels are excluded from rates and reported separately",
            "truncation": "finish_reason == 'length'",
        },
        "analysis": {
            "bootstrap_unit": "task",
            "bootstrap_iters": cfg.BOOTSTRAP_ITERS,
            "bootstrap_seed": cfg.BOOTSTRAP_SEED,
            "ci_level": cfg.CI_LEVEL,
            "comparisons": [
                "per variant x condition rates",
                "EVAL - DEPLOY gap per variant (paired over tasks)",
                "organism gap - base gap",
            ],
            "read_samples": f"random.Random({cfg.READ_SAMPLE_SEED}), 5 EVAL + 5 DEPLOY organism rows, headline arm",
        },
        "card_reference": cfg.CARD_REFERENCE,
        "environment": {
            "gpu": gpu_name(),
            "main_env": {p: importlib.metadata.version(p) for p in MAIN_PACKAGES},
            "vllm_env": vllm_env_versions(),
        },
        "outputs": {
            name: str(path.relative_to(cfg.REPO_ROOT))
            for name, path in {
                "load_report": cfg.LOAD_REPORT_PATH,
                "gate_report": cfg.GATE_REPORT_PATH,
                "generations": cfg.GENERATIONS_PATH,
                "summary": cfg.SUMMARY_PATH,
                "read_samples": cfg.READ_SAMPLES_PATH,
                "figure": cfg.FIGURE_PATH,
            }.items()
        },
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {cfg.MANIFEST_PATH} at commit {manifest['git_commit'][:7]}")


if __name__ == "__main__":
    main()
