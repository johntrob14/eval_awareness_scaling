"""exp02: time judge engine variants on the same real calls (one variant per process).

    scripts/vllm_python.sh scripts/bench_judge_engines.py --variant fp8

The calls are every production call (main variant) for 40 random Fortress questions of Qwen3-8B,
in production order, so Call B prefix reuse across a question's rollouts is exercised.
Prints calls/s and input/output tok/s; nothing is written to the results.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import ladder_config as cfg
import ladder_judge as lj

BENCH_MODEL = "Qwen3-8B"
BENCH_QUESTIONS = 40
VARIANTS = {
    "fp8": {"weights": "fp8", "prefix_caching": True, "engine": {}},
    "fp8_no_prefix_cache": {"weights": "fp8", "prefix_caching": False, "engine": {}},
    "fp8_triton_gdn": {"weights": "fp8", "prefix_caching": True, "engine": {"gdn_prefill_backend": "triton"}},
    "bf16": {"weights": "bf16", "prefix_caching": True, "engine": {}},
    "bf16_no_prefix_cache": {"weights": "bf16", "prefix_caching": False, "engine": {}},
    "bf16_triton_gdn": {"weights": "bf16", "prefix_caching": True, "engine": {"gdn_prefill_backend": "triton"}},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    variant = VARIANTS[parser.parse_args().variant]

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    from vllm import LLM, TokensPrompt

    model = cfg.JUDGE_MODEL if variant["weights"] == "bf16" else cfg.JUDGE_MODEL_PRECISION_CHECK
    path = cfg.model_dir(model["repo"], model["revision"])
    llm = LLM(model=str(path), tokenizer=str(path), seed=cfg.VLLM_ENGINE_SEED, max_model_len=cfg.JUDGE_MAX_MODEL_LEN,
              gpu_memory_utilization=cfg.VLLM_GPU_MEMORY_UTILIZATION, max_num_seqs=cfg.VLLM_MAX_NUM_SEQS,
              generation_config="vllm", enable_prefix_caching=variant["prefix_caching"],
              limit_mm_per_prompt={"image": 0, "video": 0}, **variant["engine"])
    tokenizer = llm.get_tokenizer()

    rollouts = lj.load_rollouts(BENCH_MODEL, smoke=False)
    adversarial = sorted({r["item_id"] for r in rollouts if r["prompt_set"] == "fortress_adv"})
    chosen = set(random.Random(0).sample(adversarial, BENCH_QUESTIONS))
    rollouts = [r for r in rollouts if r["item_id"] in chosen]
    calls = [c for c in lj.build_calls(BENCH_MODEL, rollouts, lj.prompt_texts(), tokenizer) if c["variant"] == "main"]
    calls.sort(key=lambda c: c["sort_key"])
    prompts = [TokensPrompt(prompt_token_ids=lj.render(tokenizer, c["message"], False)) for c in calls]
    params = [lj.sampling_for(c["call"], c["seed"], thinking=False) for c in calls]

    llm.generate(prompts[:8], params[:8], use_tqdm=False)  # warm-up
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    dt = time.time() - t0
    n_in = sum(len(p["prompt_token_ids"]) for p in prompts)
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    n_b = sum(c["call"] == "B" for c in calls)
    print(f"RESULT {parser.parse_args().variant:22s} {len(calls)} calls ({n_b} B)  {len(calls) / dt:6.2f} calls/s  "
          f"input {n_in / dt:8.0f} tok/s  output {n_out / dt:7.0f} tok/s  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
