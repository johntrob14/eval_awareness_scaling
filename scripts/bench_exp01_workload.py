"""Time vLLM on exp01's recorded generation workload, for hardware/precision/model-size planning.

Every request of one exp01 variant is replayed with its recorded output length
forced (min_tokens = max_tokens = recorded length, EOS ignored) and the arm's
sampling settings, so each run does exactly the same token work as the original
A100 run (log/exp01/generate_<variant>.log) regardless of what the model writes.
The generated text is discarded: this measures speed only.

    scripts/vllm_python.sh scripts/bench_exp01_workload.py \
        --model /workspace/models/base-text --quantization none --out log/bench/<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import exp01_config as cfg


def load_workload(variant: str) -> dict[str, list[tuple[list[int], int]]]:
    """Per arm: (prompt token IDs, recorded completion length) for each recorded completion."""
    prompt_ids = {
        p["sample_id"]: p["prompt_token_ids"]
        for p in map(json.loads, cfg.PROMPTS_PATH.open())
    }
    workload: dict[str, list[tuple[list[int], int]]] = {arm: [] for arm in cfg.ARMS}
    for row in map(json.loads, cfg.GENERATIONS_PATH.open()):
        if row["model_variant"] == variant:
            workload[row["arm"]].append(
                (prompt_ids[row["sample_id"]], row["n_generated_tokens"])
            )
    return workload


def forced_length_params(arm: dict, n_tokens: int):
    from vllm import SamplingParams

    extra = {k: arm[k] for k in ("top_p", "top_k", "min_p") if k in arm}
    return SamplingParams(
        n=1,
        temperature=arm["temperature"],
        max_tokens=n_tokens,
        min_tokens=n_tokens,
        ignore_eos=True,
        **extra,
    )


def main() -> None:
    from vllm import LLM, TokensPrompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local dir or HF repo id")
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument(
        "--workload-variant", choices=list(cfg.MODEL_VARIANTS), default="base"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; not overwriting")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    workload = load_workload(args.workload_variant)
    started = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization=None if args.quantization == "none" else args.quantization,
        seed=cfg.VLLM_ENGINE_SEED,
        max_model_len=cfg.VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config="vllm",
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    engine_startup_s = time.time() - started

    arms = {}
    for arm_name, requests in workload.items():
        arm = cfg.ARMS[arm_name]
        started = time.time()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids, _ in requests],
            [forced_length_params(arm, n) for _, n in requests],
        )
        wall_s = time.time() - started
        generated = [len(o.outputs[0].token_ids) for o in outputs]
        expected = [n for _, n in requests]
        if generated != expected:
            raise RuntimeError(
                f"{arm_name}: generated lengths differ from the workload"
            )
        arms[arm_name] = {
            "n_requests": len(requests),
            "n_generated_tokens": sum(generated),
            "max_request_tokens": max(generated),
            "wall_s": wall_s,
            "tokens_per_s": sum(generated) / wall_s,
        }
        print(
            f"{arm_name}: {len(requests)} requests, {sum(generated)} tokens in {wall_s:.0f}s",
            flush=True,
        )

    report = {
        "model": args.model,
        "quantization": args.quantization,
        "workload_variant": args.workload_variant,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_startup_s": engine_startup_s,
        "arms": arms,
        "total_generation_wall_s": sum(a["wall_s"] for a in arms.values()),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}))


if __name__ == "__main__":
    main()
