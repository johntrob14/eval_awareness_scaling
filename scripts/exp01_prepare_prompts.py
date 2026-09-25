"""exp01 step 1: build the EVAL/DEPLOY pilot prompts with the reference repo's code.

Run in /venv/main:
    python scripts/exp01_prepare_prompts.py

Writes cache/exp01/prompts.jsonl (one row per task x condition, with the exact
prompt token IDs used for generation) and cache/exp01/prompt_prep.json
(dataset hashes, held-out and pilot task lists, chat-template hash, checks).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import exp01_config as cfg
import yaml
from _reference import REFERENCE_ROOT, reference_commit, use_reference_repo
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

use_reference_repo()
from src.baseline.dataset_reader import BaselineSample, load_split_samples
from src.data.benchmark_loader import DatasetBundle
from src.qwen35.rendering import (
    apply_qwen_chat_template,
    messages_for_sample,
)
from src.qwen35.runtime import render_input_ids
from src.qwen35.thinking_modes import resolve_thinking_policy
from src.shared.condition_sets import resolve_condition_set


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_token_ids(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()


def download_dataset_bundle() -> DatasetBundle:
    """Fetch the public hua_repro_v2 export at a pinned revision.

    The reference repo's resolve_dataset_bundle() only knows a local copy of this
    dataset (it has no registered HF export), so we build the same DatasetBundle
    it would return, pointing at the public files.
    """
    paths = {
        name: Path(
            hf_hub_download(
                repo_id=cfg.DATASET_REPO,
                filename=f"{cfg.DATASET_SUBDIR}/{name}",
                repo_type="dataset",
                revision=cfg.DATASET_REVISION,
                local_dir=cfg.DATASET_CACHE_DIR,
            )
        )
        for name in cfg.DATASET_FILES
    }
    return DatasetBundle(
        dataset_id=cfg.DATASET_ID,
        source="hf_public_export",
        revision=cfg.DATASET_REVISION,
        root=paths["samples.csv"].parent,
        samples_csv=paths["samples.csv"],
        splits_csv=paths["splits.csv"],
        summary_json=paths["summary.json"],
    )


def registered_dataset_hashes() -> dict[str, str]:
    """Hashes the reference repo's dataset manifest records for this dataset."""
    manifest = yaml.safe_load(
        (REFERENCE_ROOT / "manifests/datasets/dataset_manifest_v1.yaml").read_text()
    )
    entry = next(d for d in manifest["datasets"] if d["dataset_id"] == cfg.DATASET_ID)
    return dict(entry["output_hashes"])


def samples_for_condition_set(bundle: DatasetBundle) -> list[BaselineSample]:
    """Same selection as pipelines/qwen35_ladder/run.py::samples_for_condition_set."""
    condition_set = resolve_condition_set(cfg.CONDITION_SET_ID)
    if (condition_set.dataset_id, condition_set.split_id) != (
        cfg.DATASET_ID,
        cfg.SPLIT_ID,
    ):
        raise ValueError(
            f"Condition set {cfg.CONDITION_SET_ID} is not on {cfg.DATASET_ID}"
        )
    wanted = {condition.condition_id for condition in condition_set.conditions}
    samples = [
        s for s in load_split_samples(bundle, cfg.SPLIT_ID) if s.condition_id in wanted
    ]
    if not samples:
        raise RuntimeError("No samples selected")
    return samples


def heldout_task_ids(samples: list[BaselineSample]) -> tuple[list[str], int]:
    """First HELDOUT_N_TASKS sorted task IDs that have every condition of the set.

    This mirrors how the card's held-out n=100 was drawn (load_balanced_hua_samples
    with max_tasks=100); the card does not ship its task list, so it is inferred.
    """
    all_conditions = {
        condition.condition_id
        for condition in resolve_condition_set(cfg.CONDITION_SET_ID).conditions
    }
    conditions_by_task: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        conditions_by_task[sample.task_id].add(sample.condition_id)
    complete = sorted(
        task for task, conds in conditions_by_task.items() if conds == all_conditions
    )
    return complete[: cfg.HELDOUT_N_TASKS], len(complete)


def check_special_tokens(tokenizer) -> dict[str, object]:
    """The cot/answer split and stop rule depend on these IDs; fail loudly if they differ."""
    ids = {
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "<think>": tokenizer.convert_tokens_to_ids("<think>"),
        "</think>": tokenizer.convert_tokens_to_ids("</think>"),
    }
    expected = {
        "<|im_end|>": cfg.IM_END_TOKEN_ID,
        "<think>": cfg.THINK_START_TOKEN_ID,
        "</think>": cfg.THINK_END_TOKEN_ID,
    }
    if ids != expected:
        raise ValueError(f"Special token IDs changed: {ids} != {expected}")
    think_end_is_special = cfg.THINK_END_TOKEN_ID in tokenizer.all_special_ids
    if think_end_is_special:
        raise ValueError(
            "</think> is a special token; skip_special_tokens decoding would drop it"
        )
    return {"token_ids": ids, "think_end_is_special_token": think_end_is_special}


def main() -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = download_dataset_bundle()
    file_hashes = {name: sha256_file(bundle.root / name) for name in cfg.DATASET_FILES}
    samples = samples_for_condition_set(bundle)
    heldout, n_complete_tasks = heldout_task_ids(samples)
    pilot = sorted(
        random.Random(cfg.TASK_SAMPLE_SEED).sample(heldout, cfg.PILOT_N_TASKS)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.BASE_MODEL, revision=cfg.BASE_REVISION
    )
    token_checks = check_special_tokens(tokenizer)
    policy = resolve_thinking_policy(cfg.THINKING_MODE)
    expected_prompt_suffix = "<|im_start|>assistant\n<think>\n"

    by_key = {(s.task_id, s.condition_id): s for s in samples}
    rows = []
    for task_id in pilot:
        for condition, condition_id in cfg.CONDITIONS.items():
            sample = by_key[(task_id, condition_id)]
            messages = messages_for_sample(sample, policy=policy, rewrite_identity=True)
            prompt_text = apply_qwen_chat_template(
                tokenizer, messages, policy=policy, tokenize=False
            )
            token_ids = render_input_ids(
                tokenizer, messages, policy=policy, device="cpu"
            )[0].tolist()
            if not prompt_text.endswith(expected_prompt_suffix):
                raise ValueError(
                    f"{sample.sample_id}: prompt does not end with {expected_prompt_suffix!r}"
                )
            if tokenizer.encode(prompt_text, add_special_tokens=False) != token_ids:
                raise ValueError(
                    f"{sample.sample_id}: text and token-ID renderings disagree"
                )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "task_id": task_id,
                    "condition": condition,
                    "condition_id": condition_id,
                    "context_label": sample.context_label,
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "prompt_text": prompt_text,
                    "prompt_token_ids": token_ids,
                    "prompt_token_sha256": sha256_token_ids(token_ids),
                    "n_prompt_tokens": len(token_ids),
                }
            )

    with cfg.PROMPTS_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    prep = {
        "reference_repo_commit": reference_commit(),
        "dataset": {
            "repo": cfg.DATASET_REPO,
            "revision": cfg.DATASET_REVISION,
            "subdir": cfg.DATASET_SUBDIR,
            "file_sha256": file_hashes,
            "registered_sha256_in_reference_manifest": registered_dataset_hashes(),
            "note": (
                "The public export was anonymized before upload, so its hashes differ from the "
                "reference manifest. Known text change: BigCodeBench/905 has a path replaced by "
                "'<local-path-redacted>'; that task is outside the held-out set."
            ),
            "n_samples_in_condition_set": len(samples),
            "n_complete_tasks": n_complete_tasks,
        },
        "heldout_rule": "first 100 string-sorted task IDs having all conditions of the condition set",
        "heldout_task_ids": heldout,
        "pilot_rule": f"sorted(random.Random({cfg.TASK_SAMPLE_SEED}).sample(heldout, {cfg.PILOT_N_TASKS}))",
        "pilot_task_ids": pilot,
        "redacted_prompts_in_pilot": sum("redacted" in r["prompt_text"] for r in rows),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "thinking_policy": dataclasses.asdict(policy),
        "special_token_checks": token_checks,
        "expected_prompt_suffix": expected_prompt_suffix,
        "n_prompts": len(rows),
        "prompt_token_lengths": {
            "min": min(r["n_prompt_tokens"] for r in rows),
            "max": max(r["n_prompt_tokens"] for r in rows),
        },
        "prompts_file_sha256": sha256_file(cfg.PROMPTS_PATH),
    }
    cfg.PROMPT_PREP_PATH.write_text(json.dumps(prep, indent=2, default=str) + "\n")

    print(f"{len(rows)} prompts -> {cfg.PROMPTS_PATH}")
    print(
        f"complete tasks: {n_complete_tasks}; held-out: {heldout[0]} .. {heldout[-1]}"
    )
    print(f"pilot tasks ({len(pilot)}): {pilot[:5]} ...")
    print(
        f"prompt tokens: {prep['prompt_token_lengths']}; redacted prompts in pilot: {prep['redacted_prompts_in_pilot']}"
    )
    print(f"chat template sha256: {prep['chat_template_sha256']}")
    print("dataset sha256 (public vs registered):")
    for name, digest in file_hashes.items():
        registered = prep["dataset"]["registered_sha256_in_reference_manifest"].get(
            name.replace(".csv", "_sha256").replace(".json", "_sha256")
        )
        print(f"  {name}: {digest[:16]} vs {str(registered)[:16]}")


if __name__ == "__main__":
    main()
