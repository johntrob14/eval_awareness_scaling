"""LLM-judge rollouts for verbalized eval awareness (VEA), refusal, and VEA category.

Backends:
  openai  GPT-5-mini (the post's judge). Needs OPENAI_API_KEY; set OPENAI_BASE_URL
          to use an OpenAI-compatible provider such as OpenRouter.
  mlx     a local model on Apple Silicon (default Qwen3-8B 4-bit, thinking disabled, greedy).
  vllm    a local model on NVIDIA GPUs (default Qwen3-8B, thinking disabled, greedy).

    uv run python -m eval_awareness.judge data/rollouts/Qwen3-0.6B.jsonl --task vea --backend mlx
    uv run python -m eval_awareness.judge data/rollouts/Qwen3-0.6B.jsonl --task refusal --backend openai

Judgments are written to data/judgments/<rollouts>.<task>.<judge>.jsonl and are resumable.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from eval_awareness import rubrics
from eval_awareness.backends import add_vllm_args, load_vllm
from eval_awareness.data import load_fortress
from eval_awareness.io import append_jsonl, read_jsonl

DEFAULT_JUDGE = {"openai": "gpt-5-mini", "mlx": "mlx-community/Qwen3-8B-4bit", "vllm": "Qwen/Qwen3-8B"}
DEFAULT_CHUNK = {"openai": 32, "mlx": 32, "vllm": 1024}


def judgment_path(rollouts: Path, task: str, judge: str) -> Path:
    return Path("data/judgments") / f"{rollouts.stem}.{task}.{judge.split('/')[-1]}.jsonl"


class OpenAIJudge:
    def __init__(self, model: str, workers: int = 16):
        from openai import OpenAI

        self.client, self.model, self.workers = OpenAI(), model, workers

    def _one(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content or ""

    def __call__(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, prompts))


class MLXJudge:
    def __init__(self, model: str, max_tokens: int = 400):
        from mlx_lm import load

        self.model, self.tok = load(model)
        self.max_tokens = max_tokens

    def __call__(self, prompts: list[str]) -> list[str]:
        import mlx.core as mx
        from mlx_lm import batch_generate

        toks = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False
            )
            for p in prompts
        ]
        out = batch_generate(self.model, self.tok, toks, max_tokens=self.max_tokens,
                             completion_batch_size=8, prefill_batch_size=4).texts
        mx.clear_cache()
        return out


class VLLMJudge:
    def __init__(self, model: str, args, max_tokens: int = 400):
        from vllm import SamplingParams

        self.llm = load_vllm(model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0, max_tokens=max_tokens)

    def __call__(self, prompts: list[str]) -> list[str]:
        texts = [
            self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
            for p in prompts
        ]
        return [o.outputs[0].text for o in self.llm.generate(texts, self.params)]


def build_items(task: str, rollouts: list[dict], prompts: dict[int, str], args) -> list[tuple[dict, str | None]]:
    """Return (record_stub, judge_prompt) pairs; a None prompt means decided without the judge."""
    items = []
    if task == "vea":
        for r in rollouts:
            stub = {"qid": r["qid"], "sample_idx": r["sample_idx"]}
            if args.prefilter and not rubrics.PREFILTER_RE.search(r["cot"]):
                items.append(({**stub, "vea": False, "quote": None, "prefiltered": True}, None))
            else:
                items.append((stub, rubrics.VEA_RUBRIC.format(cot=r["cot"])))
    elif task == "refusal":
        for r in rollouts:
            stub = {"qid": r["qid"], "sample_idx": r["sample_idx"]}
            if r["response"] is None:  # thinking hit max_tokens: no final response to grade
                items.append(({**stub, "refusal": None}, None))
            else:
                resp = r["response"][: args.max_response_chars]
                items.append((stub, rubrics.REFUSAL_RUBRIC + rubrics.REFUSAL_INPUTS.format(
                    prompt=prompts[r["qid"]], response=resp)))
    elif task == "category":
        vea = [j for j in read_jsonl(args.vea_judgments) if j.get("vea")]
        for j in vea:
            stub = {"qid": j["qid"], "sample_idx": j["sample_idx"], "quote": j["quote"]}
            items.append((stub, rubrics.CATEGORY_RUBRIC.format(
                prompt=prompts[j["qid"]][:1500], quote=j["quote"])))
    return items


def parse(task: str, raw: str) -> dict:
    if task == "vea":
        vea, quote = rubrics.parse_vea(raw)
        return {"vea": vea, "quote": quote}
    if task == "refusal":
        return {"refusal": rubrics.parse_yes_no(raw)}
    cat, sub = rubrics.parse_category(raw)
    return {"category": cat, "subcategory": sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", type=Path)
    ap.add_argument("--task", choices=["vea", "refusal", "category"], required=True)
    ap.add_argument("--backend", choices=["openai", "mlx", "vllm"], default="openai")
    ap.add_argument("--judge-model")
    ap.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                    help="send every CoT to the VEA judge instead of only regex hits")
    ap.add_argument("--max-response-chars", type=int, default=6000)
    ap.add_argument("--vea-judgments", type=Path, help="for --task category")
    ap.add_argument("--chunk", type=int, help="judgments per call/write (default depends on backend)")
    add_vllm_args(ap)
    args = ap.parse_args()
    args.chunk = args.chunk or DEFAULT_CHUNK[args.backend]

    judge_model = args.judge_model or DEFAULT_JUDGE[args.backend]
    if args.task == "category" and not args.vea_judgments:
        args.vea_judgments = judgment_path(args.rollouts, "vea", judge_model)
    out = judgment_path(args.rollouts, args.task, judge_model)
    done = {(j["qid"], j["sample_idx"]) for j in read_jsonl(out)}

    prompts = dict(zip(*load_fortress(exclude_explicit_eval=False)[["qid", "adversarial_prompt"]].T.values))
    items = [it for it in build_items(args.task, read_jsonl(args.rollouts), prompts, args)
             if (it[0]["qid"], it[0]["sample_idx"]) not in done]
    decided = [stub | {"judge": None} for stub, p in items if p is None]
    todo = [(stub, p) for stub, p in items if p is not None]
    append_jsonl(out, decided)
    print(f"{out}: {len(done)} existing, {len(decided)} decided without judge, {len(todo)} to judge")
    if not todo:
        return

    judge = {"openai": lambda: OpenAIJudge(judge_model), "mlx": lambda: MLXJudge(judge_model),
             "vllm": lambda: VLLMJudge(judge_model, args)}[args.backend]()
    for i in tqdm(range(0, len(todo), args.chunk)):
        chunk = todo[i: i + args.chunk]
        raws = judge([p for _, p in chunk])
        append_jsonl(out, [stub | parse(args.task, raw) | {"judge": judge_model, "raw": raw}
                           for (stub, _), raw in zip(chunk, raws)])


if __name__ == "__main__":
    main()
