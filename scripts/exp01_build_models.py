"""exp01 step 2: build the two model variants that vLLM will serve.

Run in /venv/main:
    python scripts/exp01_build_models.py

  base      Qwen3.5-4B loaded by the reference repo's HF loader and saved as-is
            (text-only), so the control goes through the same load/save path.
  organism  The same model with the 4-adapter stack merged sequentially by the
            reference repo's apply_peft_adapter_stack(), then saved.

Writes /workspace/models/{base-text,organism-merged} and
results/exp01_organism_gap/load_report.json with: adapter checksums, LoRA
tensors that matched no module (expected 144/0/0/0), per-adapter weight
changes, tied-embedding state, a bitwise reload check, and file hashes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import exp01_config as cfg
import torch
import yaml
from _reference import use_reference_repo
from safetensors import safe_open
from transformers import AutoModelForCausalLM

use_reference_repo()
from src.qwen35.adapter_stack import (
    apply_peft_adapter_stack,
    resolve_adapter_stack_paths,
    snapshot_adapter_stack_repo,
)
from src.qwen35.checkpoint_resolver import (
    CHECKPOINTS_PATH,
    resolve_qwen35_checkpoint,
)
from src.qwen35.runtime import load_qwen35_model, qwen35_runtime_status

LORA_KEY_PREFIX = "base_model.model."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_registry_matches_config(base_ckpt, organism_ckpt) -> None:
    """For an adapter stack the resolver returns the base it loads onto; the
    adapter repo/revision are only in the raw registry entry."""
    registry = yaml.safe_load(CHECKPOINTS_PATH.read_text())["checkpoints"]
    organism_entry = next(
        e for e in registry if e["checkpoint_id"] == cfg.ORGANISM_CHECKPOINT_ID
    )
    problems = []
    for ckpt in (base_ckpt, organism_ckpt):
        if (ckpt.hf_repo, ckpt.hf_revision) != (cfg.BASE_MODEL, cfg.BASE_REVISION):
            problems.append(
                f"{ckpt.checkpoint_id} base: {ckpt.hf_repo}@{ckpt.hf_revision}"
            )
    if (organism_entry["hf_repo"], organism_entry["hf_revision"]) != (
        cfg.ADAPTER_REPO,
        cfg.ADAPTER_REVISION,
    ):
        problems.append(
            f"adapter stack: {organism_entry['hf_repo']}@{organism_entry['hf_revision']}"
        )
    if [entry.path for entry in organism_ckpt.adapter_stack] != cfg.ADAPTER_ORDER:
        problems.append(
            f"adapter order: {[e.path for e in organism_ckpt.adapter_stack]}"
        )
    if not all(entry.merge_before_next for entry in organism_ckpt.adapter_stack):
        problems.append("registry does not merge every adapter before the next")
    if problems:
        raise ValueError(f"Reference registry disagrees with exp01_config: {problems}")


def verify_adapter_checksums(stack_root: Path) -> dict[str, str]:
    """Check each adapter's weights/config against the sha256 published in organism_stack.json."""
    published = json.loads((stack_root / "organism_stack.json").read_text())["files"]
    verified = {}
    for name in cfg.ADAPTER_ORDER:
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            relpath = f"{name}/{filename}"
            actual = sha256_file(stack_root / relpath)
            if actual != published[relpath]["sha256"]:
                raise ValueError(f"Checksum mismatch for {relpath}")
            verified[relpath] = actual
    return verified


def dropped_lora_tensors(model, adapter_dir: Path) -> tuple[int, list[str]]:
    """LoRA tensors in the adapter file whose target module does not exist in the model.

    PEFT silently skips these when loading, so they are counted here independently.
    """
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as handle:
        keys = list(handle.keys())
    dropped = []
    for key in keys:
        module_path = key.removeprefix(LORA_KEY_PREFIX).rsplit(".lora_", 1)[0]
        try:
            model.get_submodule(module_path)
        except AttributeError:
            dropped.append(key)
    return len(keys), dropped


def snapshot_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def parameter_group(name: str) -> str:
    """Collapse layer indices so e.g. all in_proj_qkv weights share one group."""
    return re.sub(r"\.layers\.\d+\.", ".layers.*.", name)


def weight_changes(before: dict[str, torch.Tensor], model) -> dict[str, object]:
    """Relative Frobenius change ||after - before|| / ||before||, summarised per parameter group."""
    after = model.state_dict()
    if set(after) != set(before):
        raise ValueError(
            f"State-dict keys changed: {sorted(set(after) ^ set(before))[:10]}"
        )
    groups: dict[str, dict[str, float]] = {}
    for name, old in before.items():
        new = after[name]
        relative = (
            (new.float() - old.float()).norm() / old.float().norm().clamp_min(1e-12)
        ).item()
        group = groups.setdefault(
            parameter_group(name),
            {"n_tensors": 0, "n_changed": 0, "max_relative_change": 0.0},
        )
        group["n_tensors"] += 1
        group["n_changed"] += int(not torch.equal(new, old))
        group["max_relative_change"] = max(group["max_relative_change"], relative)
    changed = {g: v for g, v in sorted(groups.items()) if v["n_changed"]}
    unchanged = sorted(g for g, v in groups.items() if not v["n_changed"])
    return {"changed_groups": changed, "unchanged_groups": unchanged}


def tie_state(model) -> dict[str, bool]:
    lm_head = model.get_output_embeddings().weight
    embed = model.get_input_embeddings().weight
    return {
        "config_tie_word_embeddings": bool(
            getattr(model.config, "tie_word_embeddings", False)
        ),
        "lm_head_shares_storage_with_embed_tokens": lm_head.data_ptr()
        == embed.data_ptr(),
        "lm_head_equals_embed_tokens": bool(torch.equal(lm_head, embed)),
    }


def save_variant(model, tokenizer, variant: str) -> Path:
    out_dir = cfg.MODEL_VARIANTS[variant]
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; delete it deliberately to rebuild")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def bitwise_reload_check(
    out_dir: Path, expected: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Reload the saved checkpoint with HF and compare every tensor to the in-memory weights."""
    reloaded = AutoModelForCausalLM.from_pretrained(
        out_dir, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    state = reloaded.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    mismatched = [
        name
        for name in expected
        if name in state and not torch.equal(state[name], expected[name].cpu())
    ]
    result = {
        "all_tensors_bitwise_equal": not (missing or extra or mismatched),
        "n_tensors": len(expected),
        "missing": missing[:10],
        "extra": extra[:10],
        "mismatched": mismatched[:10],
        "reloaded_tie_state": tie_state(reloaded),
    }
    del reloaded, state
    return result


def describe_saved(out_dir: Path) -> dict[str, object]:
    config = json.loads((out_dir / "config.json").read_text())
    return {
        "path": str(out_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "file_sha256": {
            p.name: sha256_file(p) for p in sorted(out_dir.glob("*.safetensors"))
        },
        "chat_template_sha256": sha256_file(out_dir / "chat_template.jinja")
        if (out_dir / "chat_template.jinja").exists()
        else None,
    }


def main() -> None:
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.LOAD_REPORT_PATH.exists():
        raise FileExistsError(
            f"{cfg.LOAD_REPORT_PATH} exists; results are never overwritten"
        )

    base_ckpt = resolve_qwen35_checkpoint(cfg.BASE_CHECKPOINT_ID)
    organism_ckpt = resolve_qwen35_checkpoint(cfg.ORGANISM_CHECKPOINT_ID)
    check_registry_matches_config(base_ckpt, organism_ckpt)
    stack_root = snapshot_adapter_stack_repo(cfg.ADAPTER_REPO, cfg.ADAPTER_REVISION)
    adapter_checksums = verify_adapter_checksums(stack_root)
    adapter_paths = resolve_adapter_stack_paths(
        organism_ckpt.adapter_stack, local_root=str(stack_root)
    )

    _, tokenizer, model = load_qwen35_model(base_ckpt, device="cuda", dtype="bfloat16")
    report: dict[str, object] = {
        "hf_runtime_status": dataclasses.asdict(qwen35_runtime_status()),
        "model_class": type(model).__name__,
        "adapter_checksums_verified_against_organism_stack_json": adapter_checksums,
        "base_tie_state": tie_state(model),
    }

    base_state = snapshot_state(model)
    save_variant(model, tokenizer, "base")
    print("saved base-text")

    per_adapter = []
    for entry, adapter_path in zip(
        organism_ckpt.adapter_stack, adapter_paths, strict=True
    ):
        n_tensors, dropped = dropped_lora_tensors(model, adapter_path)
        before = snapshot_state(model)
        model = apply_peft_adapter_stack(model, [adapter_path])
        per_adapter.append(
            {
                "adapter": entry.path,
                "n_lora_tensors": n_tensors,
                "n_dropped_lora_tensors": len(dropped),
                "expected_dropped": cfg.EXPECTED_DROPPED_LORA_TENSORS[entry.path],
                "dropped_module_kinds": sorted(
                    {k.rsplit(".lora_", 1)[0].split(".")[-1] for k in dropped}
                ),
                "weight_changes": weight_changes(before, model),
                "tie_state_after": tie_state(model),
            }
        )
        del before
        print(f"merged {entry.path}: dropped {len(dropped)}/{n_tensors} LoRA tensors")
    report["per_adapter"] = per_adapter
    report["organism_vs_base"] = weight_changes(base_state, model)
    report["organism_tie_state"] = tie_state(model)

    organism_state = snapshot_state(model)
    save_variant(model, tokenizer, "organism")
    print("saved organism-merged")
    del model
    torch.cuda.empty_cache()

    report["bitwise_reload_check"] = {
        "base": bitwise_reload_check(cfg.MODEL_VARIANTS["base"], base_state),
        "organism": bitwise_reload_check(
            cfg.MODEL_VARIANTS["organism"], organism_state
        ),
    }
    report["saved"] = {
        variant: describe_saved(path) for variant, path in cfg.MODEL_VARIANTS.items()
    }

    dropped_ok = all(
        a["n_dropped_lora_tensors"] == a["expected_dropped"] for a in per_adapter
    )
    reload_ok = all(
        r["all_tensors_bitwise_equal"] for r in report["bitwise_reload_check"].values()
    )
    report["checks"] = {
        "dropped_counts_as_expected": dropped_ok,
        "bitwise_reload_ok": reload_ok,
    }
    cfg.LOAD_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["checks"]))
    if not (dropped_ok and reload_ok):
        raise SystemExit(f"Build checks failed; see {cfg.LOAD_REPORT_PATH}")


if __name__ == "__main__":
    main()
