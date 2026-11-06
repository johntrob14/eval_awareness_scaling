"""Make one of our quantized checkpoints with llm-compressor (separate env: envs/quantize).

    UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/quantize uv run --project envs/quantize python src/eval_awareness/quant/quantize.py \
        --variant Qwen3.5-4B@gptq-w4

Recipe (config.RECIPES): GPTQ W4A16 / W8A16, group 128, symmetric, static act-order (the scheme's defaults),
dampening 0.05, calibrated on 1,024 Open-Platypus rows as raw instruction text truncated to 8,192 tokens; or RTN
W4A16 with no calibration. Kept in bf16: lm_head, embeddings, and for Qwen3.5 the Gated-DeltaNet linear attention,
vision tower and MTP heads (config.IGNORE). This is RedHat's published Qwen3.5 recipe, applied to every base.
Writes QUANT_CHECKPOINTS/<variant>/ with recipe_provenance.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_awareness.quant import config as qc  # noqa: E402


def calibration_dataset(tokenizer, lo: int, hi: int):
    from datasets import load_dataset

    c = qc.CALIBRATION
    ds = load_dataset(c["repo"], revision=c["revision"], split=f"train[{lo}:{hi}]").shuffle(seed=42)

    def text(ex):
        t = ex["instruction"]
        if ex.get("input"):
            t += "\n" + ex["input"]
        return {"text": t}

    def tokenize(ex):
        return tokenizer(ex["text"], padding=False, max_length=c["max_seq_len"], truncation=True,
                         add_special_tokens=False)

    return ds.map(text).map(tokenize, remove_columns=ds.column_names + ["text"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="<base>@<recipe>, recipe one of " + ", ".join(qc.RECIPES))
    args = ap.parse_args()
    base, condition = qc.split_variant(args.variant)
    if condition not in qc.RECIPES:
        raise SystemExit(f"{condition} is not one of our recipes ({list(qc.RECIPES)})")
    recipe_cfg = qc.RECIPES[condition]
    out = qc.checkpoint_dir(base, condition)
    if (out / "recipe_provenance.json").exists():
        print(f"{out} already exists; delete it to requantize")
        return

    import llmcompressor
    import torch
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = qc.ladder_config()
    model_info = cfg.ALL_SUBJECTS[base]
    src = cfg.model_dir(model_info["repo"], model_info["revision"])
    qwen35 = model_info["family"] != "qwen3"
    t0 = time.time()
    if qwen35:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype="auto")
        layer = "Qwen3_5DecoderLayer"
    else:
        model = AutoModelForCausalLM.from_pretrained(src, dtype="auto")
        layer = "Qwen3DecoderLayer"
    tokenizer = AutoTokenizer.from_pretrained(src)

    if recipe_cfg["method"] == "gptq":
        lo, hi = recipe_cfg["calibration_slice"]
        ds = calibration_dataset(tokenizer, lo, hi)
        recipe = GPTQModifier(targets="Linear", scheme=recipe_cfg["scheme"], ignore=qc.IGNORE,
                              dampening_frac=qc.GPTQ_DAMPENING)
        # llm-compressor >= 0.11 takes sequential_targets in oneshot, not in the modifier (RedHat used 0.10)
        # text-only calibration: the tokenizer is the processor (Qwen3.5's AutoProcessor needs torchvision)
        oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=qc.CALIBRATION["max_seq_len"],
                num_calibration_samples=hi - lo, sequential_targets=[layer], processor=tokenizer)
    else:
        oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme=recipe_cfg["scheme"], ignore=qc.IGNORE),
                processor=tokenizer)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    # Tokenizer, chat template and (Qwen3.5) processor configs copied verbatim from the bf16 checkpoint.
    for f in Path(src).iterdir():
        if f.suffix in (".json", ".jinja", ".txt", ".model") and not (out / f.name).exists() \
                and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    provenance = {"variant": args.variant, "base_repo": model_info["repo"], "base_revision": model_info["revision"],
                  "recipe": recipe_cfg, "ignore": qc.IGNORE, "dampening": qc.GPTQ_DAMPENING,
                  "calibration": qc.CALIBRATION if recipe_cfg["method"] == "gptq" else None,
                  "sequential_target": layer, "seconds": time.time() - t0,
                  "llmcompressor": llmcompressor.__version__, "transformers": transformers.__version__,
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out / "recipe_provenance.json").write_text(json.dumps(provenance, indent=1))  # written last: completion marker
    print(f"wrote {out} in {provenance['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
