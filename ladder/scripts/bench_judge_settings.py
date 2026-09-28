"""exp02: time the judge's sampling settings on the same real calls (one engine load).

    scripts/vllm_python.sh scripts/bench_judge_settings.py

Variants: production (JSON schema + top-20 logprobs), schema without logprobs, logprobs without
schema, neither. Prints calls/s per variant; nothing is written to the results.
"""

from __future__ import annotations

import random
import time

import ladder_config as cfg
import ladder_judge as lj

N_CALLS = 400


def main() -> None:
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    llm = lj.make_judge(precision_check=True)  # the FP8 checkpoint, the judge when this was run
    tokenizer = llm.get_tokenizer()
    rollouts = lj.load_rollouts("Qwen3-0.6B", smoke=False)
    rollouts = random.Random(0).sample(rollouts, 3000)
    calls = [c for c in lj.build_calls("Qwen3-0.6B", rollouts, lj.prompt_texts(), tokenizer) if c["variant"] == "main"]
    calls = random.Random(1).sample(calls, N_CALLS)
    prompts = [TokensPrompt(prompt_token_ids=lj.render(tokenizer, c["message"], False)) for c in calls]

    def params(call, schema: bool, logprobs: bool):
        extra = {}
        if schema:
            extra["structured_outputs"] = StructuredOutputsParams(
                json=lj.CALL_A_SCHEMA if call["call"] == "A" else lj.CALL_B_SCHEMA, disable_any_whitespace=True)
        if logprobs:
            extra["logprobs"] = cfg.JUDGE_LOGPROBS
        return SamplingParams(n=1, seed=call["seed"], max_tokens=cfg.JUDGE_MAX_TOKENS, **cfg.JUDGE_SAMPLING_NONTHINKING, **extra)

    llm.generate(prompts[:8], [params(c, True, True) for c in calls[:8]])  # warm-up
    for name, schema, logprobs in [("schema+logprobs", True, True), ("schema only", True, False),
                                   ("logprobs only", False, True), ("neither", False, False)]:
        t0 = time.time()
        outs = llm.generate(prompts, [params(c, schema, logprobs) for c in calls], use_tqdm=False)
        dt = time.time() - t0
        n_in = sum(len(p["prompt_token_ids"]) for p in prompts)
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{name:16s} {N_CALLS / dt:6.2f} calls/s  input {n_in / dt:8.0f} tok/s  output {n_out / dt:7.0f} tok/s  "
              f"({dt:.0f}s, mean out {n_out / N_CALLS:.0f} tokens)", flush=True)


if __name__ == "__main__":
    main()
