"""Judge workload per model: calls and input tokens by call type, and the Call B tokens that prefix
caching could reuse (rubric + user request shared by all rollouts of a question). CPU only.

    /venv/main/bin/python scripts/judge_call_budget.py Qwen3-0.6B Qwen3-8B
"""
import sys
from collections import defaultdict

from transformers import AutoTokenizer

import ladder_config as cfg
import ladder_judge as judge

HYBRID_CACHE_BLOCK_TOKENS = 784  # vLLM 'align' mode block size for Qwen3.8-27B; only whole blocks are reused


def main(models: list[str]) -> None:
    path = cfg.model_dir(cfg.JUDGE_MODEL["repo"], cfg.JUDGE_MODEL["revision"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    texts = judge.prompt_texts()
    for model in models:
        calls = judge.build_calls(model, judge.load_rollouts(model, smoke=False), texts, tokenizer)
        tokens = defaultdict(int)
        counts = defaultdict(int)
        reusable = 0
        seen_questions = set()
        for call in calls:
            n = len(judge.render(tokenizer, call["message"], thinking=False))
            kind = f"{call['call']}:{call['variant']}"
            tokens[kind] += n
            counts[kind] += 1
            if call["call"] == "B":
                question = (call["prompt_set"], call["item_id"])
                if question in seen_questions:
                    prompt_end = call["message"].index("</request>")
                    prefix = len(judge.render(tokenizer, call["message"][:prompt_end], thinking=False))
                    reusable += (prefix // HYBRID_CACHE_BLOCK_TOKENS) * HYBRID_CACHE_BLOCK_TOKENS
                seen_questions.add(question)
        total = sum(tokens.values())
        print(f"\n{model}: {len(calls)} calls, {total / 1e6:.2f}M input tokens")
        for kind in sorted(tokens):
            print(f"  {kind:28s} {counts[kind]:6d} calls  {tokens[kind] / 1e6:6.2f}M tokens  "
                  f"mean {tokens[kind] / counts[kind]:6.0f}")
        print(f"  Call B tokens reusable by prefix caching: {reusable / 1e6:.2f}M ({reusable / total:.0%} of all input)")


if __name__ == "__main__":
    main(sys.argv[1:])
