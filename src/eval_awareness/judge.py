"""LLM-judge rollouts for verbalized eval awareness (VEA), refusal, and VEA category.

Backends:
  openai  GPT-5-mini (the post's judge). Needs OPENAI_API_KEY; set OPENAI_BASE_URL
          to use an OpenAI-compatible provider such as OpenRouter.
  mlx     a local model on Apple Silicon (default Qwen3-8B 4-bit, thinking disabled, greedy).
  vllm    a local model on NVIDIA GPUs (default Qwen3-8B, thinking disabled, greedy).

    uv run python -m eval_awareness.judge data/rollouts/Qwen3-0.6B.jsonl --task vea --backend mlx
    uv run python -m eval_awareness.judge data/rollouts/Qwen3-0.6B.jsonl --task refusal --backend openai

Judgments are written to data/judgments/<rollouts>.<task>.<judge>.jsonl and are resumable.
Tasks vea_response (App. F1 applied to the final response) and safety (a direct safety-relevance
tag per VEA quote) were added for the Gemma pilot; the other tasks' defaults are unchanged.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from eval_awareness import rubrics
from eval_awareness.backends import add_vllm_args, load_vllm
from eval_awareness.data import load_fortress
from eval_awareness.io import append_jsonl, read_jsonl

DEFAULT_JUDGE = {"openai": "gpt-5-mini", "mlx": "mlx-community/Qwen3-8B-4bit", "vllm": "Qwen/Qwen3-8B"}
DEFAULT_CHUNK = {"openai": 32, "mlx": 32, "vllm": 1024}


def judgment_path(rollouts: Path, task: str, judge: str, out_dir: Path = Path("data/judgments")) -> Path:
    return Path(out_dir) / f"{rollouts.stem}.{task}.{judge.split('/')[-1]}.jsonl"


def key(j: dict) -> tuple:
    # CoT-VEA stubs carry no "source", so keys for pre-existing Qwen judgments are unchanged.
    return (j["qid"], j["sample_idx"]) + ((j["source"],) if "source" in j else ())


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

        extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if args.text_only else {}
        self.llm = load_vllm(model, args.tensor_parallel_size, args.gpu_memory_utilization, args.max_model_len,
                         pipeline_parallel_size=args.pipeline_parallel_size,
                             **extra)
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
    if task == "vea_response":
        for r in rollouts:
            stub = {"qid": r["qid"], "sample_idx": r["sample_idx"]}
            if not r["response"]:
                items.append(({**stub, "vea": None, "quote": None}, None))
            else:
                items.append((stub, rubrics.VEA_RESPONSE_RUBRIC.format(cot=r["response"][: args.max_vea_chars])))
    elif task == "vea":
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
    elif task in ("category", "safety"):
        for path in args.vea_judgments:
            src = {"source": "response"} if ".vea_response." in Path(path).name else {}
            for j in read_jsonl(path):
                if not j.get("vea"):
                    continue
                stub = {"qid": j["qid"], "sample_idx": j["sample_idx"], "quote": j["quote"]} | src
                items.append((stub, rubrics.CATEGORY_RUBRIC.format(prompt=prompts[j["qid"]][:1500], quote=j["quote"])
                              if task == "category" else rubrics.SAFETY_RUBRIC.format(quote=j["quote"])))
    return items


def parse(task: str, raw: str) -> dict:
    if task in ("vea", "vea_response"):
        vea, quote = rubrics.parse_vea(raw)
        return {"vea": vea, "quote": quote}
    if task == "refusal":
        return {"refusal": rubrics.parse_yes_no(raw)}
    if task == "safety":
        return {"safety": rubrics.parse_safety(raw)}
    cat, sub = rubrics.parse_category(raw)
    return {"category": cat, "subcategory": sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", type=Path)
    ap.add_argument("--task", choices=["vea", "vea_response", "refusal", "category", "safety"], required=True)
    ap.add_argument("--backend", choices=["openai", "mlx", "vllm"], default="openai")
    ap.add_argument("--judge-model")
    ap.add_argument("--no-prefilter", dest="prefilter", action="store_false",
                    help="send every CoT to the VEA judge instead of only regex hits")
    ap.add_argument("--max-response-chars", type=int, default=6000)
    ap.add_argument("--max-vea-chars", type=int, default=40000, help="for --task vea_response")
    ap.add_argument("--vea-judgments", type=Path, nargs="+",
                    help="for --task category/safety (default: this judge's CoT VEA judgments)")
    ap.add_argument("--chunk", type=int, help="judgments per call/write (default depends on backend)")
    ap.add_argument("--split", choices=["adversarial", "benign"], default="adversarial",
                    help="Fortress prompt the rollouts answered (refusal/category inputs)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/judgments"))
    ap.add_argument("--judge-max-tokens", type=int, default=400, help="vllm/mlx: judge output limit")
    ap.add_argument("--text-only", action="store_true", help="vllm: disable image/video inputs of a multimodal judge")
    add_vllm_args(ap)
    args = ap.parse_args()
    args.chunk = args.chunk or DEFAULT_CHUNK[args.backend]

    judge_model = args.judge_model or DEFAULT_JUDGE[args.backend]
    if args.task in ("category", "safety") and not args.vea_judgments:
        args.vea_judgments = [judgment_path(args.rollouts, "vea", judge_model, args.out_dir)]
    out = judgment_path(args.rollouts, args.task, judge_model, args.out_dir)
    done = {key(j) for j in read_jsonl(out)}

    col = "adversarial_prompt" if args.split == "adversarial" else "benign_prompt"
    prompts = dict(zip(*load_fortress(exclude_explicit_eval=False)[["qid", col]].T.values))
    # keep the last row per rollout (generate --retry-truncated appends replacements)
    rollouts = list({(r["qid"], r["sample_idx"]): r for r in read_jsonl(args.rollouts)}.values())
    items = [it for it in build_items(args.task, rollouts, prompts, args) if key(it[0]) not in done]
    decided = [stub | {"judge": None} for stub, p in items if p is None]
    todo = [(stub, p) for stub, p in items if p is not None]
    append_jsonl(out, decided)
    print(f"{out}: {len(done)} existing, {len(decided)} decided without judge, {len(todo)} to judge")
    if not todo:
        return

    judge = {"openai": lambda: OpenAIJudge(judge_model),
             "mlx": lambda: MLXJudge(judge_model, args.judge_max_tokens),
             "vllm": lambda: VLLMJudge(judge_model, args, args.judge_max_tokens)}[args.backend]()
    if args.backend == "vllm":  # record the exact judge config next to its outputs
        import vllm

        example = judge.tok.apply_chat_template([{"role": "user", "content": todo[0][1]}], add_generation_prompt=True,
                                                enable_thinking=False, tokenize=False)
        append_jsonl(out.with_suffix(".config.jsonl"), [{
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": args.task, "judge_model": judge_model,
            "vllm": vllm.__version__, "sampling": repr(judge.params), "enable_thinking": False,
            "tensor_parallel_size": args.tensor_parallel_size, "max_model_len": args.max_model_len,
            "n_to_judge": len(todo), "example_rendered_prompt": example,
        }])
    for i in tqdm(range(0, len(todo), args.chunk)):
        chunk = todo[i: i + args.chunk]
        raws = judge([p for _, p in chunk])
        append_jsonl(out, [stub | parse(args.task, raw) | {"judge": judge_model, "raw": raw}
                           | ({"parse_ok": "QUOTE:" in raw} if args.task in ("vea", "vea_response") else {})
                           for (stub, _), raw in zip(chunk, raws)])


if __name__ == "__main__":
    main()
