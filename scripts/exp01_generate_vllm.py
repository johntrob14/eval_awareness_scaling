"""exp01 step 4: generate both decoding arms for one model variant with vLLM.

Run in the vLLM venv, once per variant (separate processes free GPU memory):
    /workspace/.venv-vllm/bin/python scripts/exp01_generate_vllm.py --variant organism
    /workspace/.venv-vllm/bin/python scripts/exp01_generate_vllm.py --variant base

Prompts are passed as the exact token IDs from cache/exp01/prompts.jsonl.
Sampled arms use a per-prompt seed (exp01_config.prompt_seed), shared by both
variants. Output: cache/exp01/raw_generations_<variant>_<key>.jsonl with raw
token IDs, vLLM's finish_reason, and the decoded text.
"""

from __future__ import annotations

import argparse
import json
import time

import exp01_config as cfg


def sampling_params_for(arm: dict, sample_id: str):
    from vllm import SamplingParams

    sampled = arm["temperature"] > 0
    extra = {
        k: arm[k]
        for k in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
        if k in arm
    }
    return SamplingParams(
        n=arm["n"],
        temperature=arm["temperature"],
        max_tokens=arm["max_tokens"],
        seed=cfg.prompt_seed(sample_id) if sampled else None,
        stop_token_ids=[cfg.IM_END_TOKEN_ID],
        **extra,
    )


def main() -> None:
    from exp01_vllm_common import make_llm
    from vllm import TokensPrompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(cfg.MODEL_VARIANTS), required=True)
    args = parser.parse_args()

    out_path = cfg.raw_generations_path(args.variant)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} exists (same weights, prompts and arms); not regenerating"
        )
    prompts = [json.loads(line) for line in cfg.PROMPTS_PATH.open()]
    llm = make_llm(cfg.MODEL_VARIANTS[args.variant])

    rows = []
    for arm_name, arm in cfg.ARMS.items():
        started = time.time()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p in prompts],
            [sampling_params_for(arm, p["sample_id"]) for p in prompts],
        )
        for prompt, output in zip(prompts, outputs, strict=True):
            for completion in output.outputs:
                rows.append(
                    {
                        "model_variant": args.variant,
                        "arm": arm_name,
                        "sample_id": prompt["sample_id"],
                        "sample_idx": completion.index,
                        "seed": cfg.prompt_seed(prompt["sample_id"])
                        if arm["temperature"] > 0
                        else None,
                        "token_ids": list(completion.token_ids),
                        "n_generated_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "text": completion.text,
                    }
                )
        arm_rows = [r for r in rows if r["arm"] == arm_name]
        n_length = sum(r["finish_reason"] == "length" for r in arm_rows)
        print(
            f"{args.variant} {arm_name}: {len(arm_rows)} completions in {time.time() - started:.0f}s, "
            f"{sum(r['n_generated_tokens'] for r in arm_rows)} tokens, {n_length} hit max_tokens",
            flush=True,
        )

    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
