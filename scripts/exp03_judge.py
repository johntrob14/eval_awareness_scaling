"""exp03 judge: 4-way awareness label and graded refusal, label-first JSON, Qwen3.8-27B (bf16 or FP8).

Call A (reasoning) is asked only for rollouts whose CoT hits the recall regex, on the same excerpt windows as
exp02; regex-negative rollouts are awareness "none" (a 3% full-CoT audit checks that). Call B (answer) is asked
for every scored rollout on the stored answer (at most 384 tokens).

Modes (all run with scripts/vllm_python.sh):
  smoke --engine bf16|fp8|qwen35_9b   judge the smoke items (random exp02 calls + exp02's 500 validation items)
  label --models ...        relabel every rollout of the given models (engine/format from the smoke decision)
  interventions --models    Call B on the answers of the insertion/removal continuations
Outputs: cache/exp03/{smoke,judge}/*.jsonl, one row per call. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import exp03_config as cfg
import exp03_data as data
import ladder_config as base
from ladder_judge import cot_windows, parse_json, render

# --- Schemas -----------------------------------------------------------------------------------------


def call_a_schema(answer_format: str) -> dict:
    fields = {
        "mentions_evaluation": {"type": "boolean"},
        "mentions_user_probing": {"type": "boolean"},
        "mentions_manipulation": {"type": "boolean"},
        "awareness": {"type": "string", "enum": cfg.AWARENESS_TYPES},
        "framing": {"type": "string", "enum": cfg.FRAMINGS},
        "quote": {"type": ["string", "null"]},
    }
    if answer_format == "label_first":
        fields["reason"] = {"type": "string"}
    else:
        fields = {"reasoning": {"type": "string"}, **fields}
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def call_b_schema(answer_format: str) -> dict:
    fields = {"refusal": {"type": "string", "enum": cfg.REFUSAL_TYPES}}
    if answer_format == "label_first":
        fields["reason"] = {"type": "string"}
    else:
        fields = {"reasoning": {"type": "string"}, **fields}
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


BOOLEAN_FIELDS = {"A": list(cfg.MENTION_FLAGS.values()), "B": []}
ENUM_FIELDS = {"A": {"awareness": cfg.AWARENESS_TYPES}, "B": {"refusal": cfg.REFUSAL_TYPES}}


def call_a_message(cot: str, answer_format: str) -> str:
    return cfg.CALL_A_RUBRIC.format(cot=cot, answer_format=cfg.CALL_A_FORMAT[answer_format])


def call_b_message(prompt: str, answer: str, cut: bool, answer_format: str) -> str:
    note = cfg.CALL_B_CUT_NOTE.format(n=base.ANSWER_CAP_TOKENS) if cut else ""
    # str.replace, not format: user prompts and answers contain braces
    return (cfg.CALL_B_RUBRIC.replace("{answer_format}", cfg.CALL_B_FORMAT[answer_format])
            .replace("{cut_note}", note).replace("{prompt}", prompt).replace("{answer}", answer))


def resolved_awareness(labels: dict) -> str:
    """The primary label: the first mention flag that is true, in precedence order."""
    for kind in cfg.AWARENESS_TYPES[:-1]:
        if labels.get(cfg.MENTION_FLAGS[kind]):
            return kind
    return "none"


def call_seed(key: str, call: str, variant: str) -> int:
    return int(hashlib.sha256(f"exp03|{key}|{call}|{variant}".encode()).hexdigest()[:8], 16)


# --- Engine -----------------------------------------------------------------------------------------------


def judge_dir(engine: str) -> Path:
    weights = cfg.JUDGE_ENGINE_WEIGHTS[engine]
    return base.model_dir(weights["repo"], weights["revision"])


def judge_tokenizer(engine: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(judge_dir(engine)))


def make_judge(engine: str):
    """Qwen3.8-27B from the bf16 checkpoint ("fp8": vLLM quantizes it to FP8 at load time), or Qwen3.5-9B bf16."""
    if engine not in cfg.JUDGE_ENGINES:
        raise ValueError(engine)
    # DeepGEMM's FP8 kernels JIT-compile with nvcc >= 12.9 and this image has 12.8. Set before importing vLLM.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    from vllm import LLM

    path = judge_dir(engine)
    quantization = {"quantization": "fp8"} if engine == "fp8" else {}
    return LLM(model=str(path), tokenizer=str(path), seed=base.VLLM_ENGINE_SEED,
               max_model_len=base.JUDGE_MAX_MODEL_LEN, gpu_memory_utilization=base.VLLM_GPU_MEMORY_UTILIZATION,
               max_num_seqs=base.VLLM_MAX_NUM_SEQS, generation_config="vllm", enable_prefix_caching=False,
               limit_mm_per_prompt={"image": 0, "video": 0}, **quantization)


def sampling_for(call: str, seed: int, answer_format: str):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = call_a_schema(answer_format) if call == "A" else call_b_schema(answer_format)
    return SamplingParams(n=1, seed=seed, max_tokens=cfg.JUDGE_MAX_TOKENS, logprobs=cfg.JUDGE_LOGPROBS,
                          structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
                          **cfg.JUDGE_SAMPLING)


def value_token_index(pieces: list[str], field: str) -> int | None:
    """Index of the first generated token of `field`'s value in the JSON output."""
    match = re.search(rf'"{field}"\s*:\s*"?', "".join(pieces))
    if not match:
        return None
    offset = 0
    for i, piece in enumerate(pieces):
        if offset + len(piece) > match.end():
            return i
        offset += len(piece)
    return None


def boolean_probability(completion, pieces: list[str], field: str) -> float | None:
    """P(true) / (P(true) + P(false)) at the value token (raw logprobs, top 20)."""
    i = value_token_index(pieces, field)
    if i is None:
        return None
    p = {"t": 0.0, "f": 0.0}
    for lp in completion.logprobs[i].values():
        token = (lp.decoded_token or "").strip(' ":').lower()
        if token and token[0] in p and ("true".startswith(token) or "false".startswith(token)):
            p[token[0]] += math.exp(lp.logprob)
    total = p["t"] + p["f"]
    return p["t"] / total if total > 0 else None


def enum_probabilities(completion, pieces: list[str], field: str, options: list[str]) -> dict | None:
    """Probability of each enum option from the first token of the value (options have distinct first tokens)."""
    i = value_token_index(pieces, field)
    if i is None:
        return None
    mass = dict.fromkeys(options, 0.0)
    for lp in completion.logprobs[i].values():
        token = (lp.decoded_token or "").lstrip(' ":')
        if not token:
            continue
        matches = [o for o in options if o.startswith(token)]
        if len(matches) == 1:
            mass[matches[0]] += math.exp(lp.logprob)
    total = sum(mass.values())
    return {o: m / total for o, m in mass.items()} if total > 0 else None


def run_calls(llm, tokenizer, calls: list[dict], answer_format: str) -> tuple[list[dict], dict]:
    from vllm import TokensPrompt

    calls = sorted(calls, key=lambda c: c["sort_key"])
    prompts = [TokensPrompt(prompt_token_ids=render(tokenizer, c["message"], False)) for c in calls]
    start = time.time()
    outputs = llm.generate(prompts, [sampling_for(c["call"], c["seed"], answer_format) for c in calls])
    elapsed = time.time() - start
    rows = []
    for call, prompt, out in zip(calls, prompts, outputs, strict=True):
        completion = out.outputs[0]
        pieces = [tokenizer.decode([t]) for t in completion.token_ids]
        labels = parse_json(completion.text)
        row = {k: v for k, v in call.items() if k not in ("message", "sort_key")}
        row.update({"answer_format": answer_format, "parse_ok": labels is not None, "labels": labels,
                    "finish_reason": completion.finish_reason,
                    "n_input_tokens": len(prompt["prompt_token_ids"]), "n_output_tokens": len(completion.token_ids),
                    "p_true": {f: boolean_probability(completion, pieces, f) for f in BOOLEAN_FIELDS[call["call"]]},
                    "p_enum": {f: enum_probabilities(completion, pieces, f, opts)
                               for f, opts in ENUM_FIELDS[call["call"]].items()}})
        if labels is not None and call["call"] == "A":
            row["awareness"] = resolved_awareness(labels)
        if labels is not None and call["call"] == "B":
            row["refusal"] = labels["refusal"]
        rows.append(row)
    timing = {"n_calls": len(calls), "seconds": elapsed, "calls_per_s": len(calls) / elapsed,
              "mean_output_tokens": sum(r["n_output_tokens"] for r in rows) / len(rows),
              "mean_input_tokens": sum(r["n_input_tokens"] for r in rows) / len(rows)}
    return rows, timing


def write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; exp03 outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.rename(path)


def rollout_call(r: dict, call: str, variant: str, message: str, **extra) -> dict:
    key = data.rollout_key(r)
    order = (r["prompt_set"], r["item_id"], r["model"], r["stage"], r["rollout_index"])
    return {"model": r["model"], "rollout_key": key, "prompt_set": r["prompt_set"], "item_id": r["item_id"],
            "stage": r["stage"], "rollout_index": r["rollout_index"], "call": call, "variant": variant,
            "seed": call_seed(key, call, variant), "message": message, "sort_key": (call, *order), **extra}


# --- smoke ---------------------------------------------------------------------------------------------------


def smoke_items() -> list[dict]:
    """Random exp02 main calls (stratified over models) plus exp02's 500 uncertainty-weighted validation items.

    Each item carries what both answer formats need: Call A the exp02 excerpt window, Call B prompt + answer.
    """
    path = cfg.SMOKE_DIR / "judge_items.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(cfg.SMOKE_SEED)
    texts = data.prompt_texts()
    validation = json.loads((base.CACHE_DIR / "judge" / "validation_items.json").read_text())
    validation_keys = {(v["rollout_key"], v["call"]) for v in validation}
    items = []
    exp02_models = [m for m in cfg.RELABEL_MODELS if m not in cfg.GENERATED_IN_EXP03]
    for model in exp02_models:
        labels = data.exp02_main_labels(model)
        rollouts = {data.rollout_key(r): r for r in data.load_rollouts(model, exp02_only=True)}
        for call, n_total in cfg.SMOKE_RANDOM_ITEMS.items():
            pool = sorted(k for k in labels if k[1] == call and k[0] in rollouts)
            n = round(n_total / len(exp02_models))
            picked = [("random", k) for k in rng.sample(pool, min(n, len(pool)))]
            picked += [("validation", k) for k in pool if k in validation_keys]
            for source, (key, _) in picked:
                r, exp02 = rollouts[key], labels[(key, call)]
                item = {"source": source, "model": model, "rollout_key": key, "call": call,
                        "prompt_set": r["prompt_set"], "item_id": r["item_id"], "stage": r["stage"],
                        "rollout_index": r["rollout_index"], "exp02_labels": exp02["labels"]}
                if call == "A":
                    item["cot"] = exp02["window_text"]
                else:
                    item.update(prompt=texts[(r["prompt_set"], r["item_id"])], answer=r["answer_text"],
                                answer_cut=r["answer_cut"])
                items.append(item)
    # an item drawn at random and also in the validation set is kept once, as "random"
    seen, unique = set(), []
    for item in sorted(items, key=lambda i: i["source"] != "random"):
        if (item["rollout_key"], item["call"]) not in seen:
            seen.add((item["rollout_key"], item["call"]))
            unique.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(unique))
    return unique


# bf16 runs label-first twice with different seeds (the noise floor) and reasoning-first once; FP8 runs
# label-first with the second seed set, so FP8-vs-bf16(rep1) has the same seed structure as bf16 rep2-vs-rep1.
# The Qwen3.5-9B candidate is compared in the chosen reasoning-first format, so bf16 also gets a second
# reasoning-first pass: that format's own noise floor.
SMOKE_PASSES = {"bf16": [("label_first", "rep1"), ("label_first", "rep2"), ("reasoning_first", "rep1"),
                         ("reasoning_first", "rep2")],
                "fp8": [("label_first", "rep2")],
                "qwen35_9b": [("reasoning_first", "rep1"), ("reasoning_first", "rep2")]}


def smoke_pass_path(engine: str, answer_format: str, rep: str) -> Path:
    return cfg.SMOKE_DIR / f"judge_{engine}_{answer_format}_{rep}.jsonl"


def run_smoke(engine: str) -> None:
    items = smoke_items()
    passes = [(f, rep) for f, rep in SMOKE_PASSES[engine] if not smoke_pass_path(engine, f, rep).exists()]
    if not passes:
        print(f"[smoke] {engine}: all passes exist")
        return
    llm = make_judge(engine)
    tokenizer = judge_tokenizer(engine)
    for answer_format, rep in passes:
        calls = []
        for i, item in enumerate(items):
            message = (call_a_message(item["cot"], answer_format) if item["call"] == "A" else
                       call_b_message(item["prompt"], item["answer"], item["answer_cut"], answer_format))
            calls.append({"item": i, "source": item["source"], "model": item["model"],
                          "rollout_key": item["rollout_key"], "call": item["call"], "variant": rep,
                          "seed": call_seed(item["rollout_key"], item["call"], f"smoke|{rep}"),
                          "message": message, "sort_key": (item["call"], i)})
        rows, timing = run_calls(llm, tokenizer, calls, answer_format)
        write_new(smoke_pass_path(engine, answer_format, rep), rows)
        timing.update(engine=engine, answer_format=answer_format, rep=rep)
        (cfg.SMOKE_DIR / f"judge_{engine}_{answer_format}_{rep}_timing.json").write_text(json.dumps(timing, indent=1))
        print(f"[smoke] {engine} {answer_format} {rep}: {timing['calls_per_s']:.2f} calls/s, "
              f"mean output {timing['mean_output_tokens']:.0f} tokens", flush=True)


# --- label ------------------------------------------------------------------------------------------------


def judge_decision() -> dict:
    if not cfg.JUDGE_ENGINE_DECISION_PATH.exists():
        raise FileNotFoundError(f"{cfg.JUDGE_ENGINE_DECISION_PATH} missing; run the smoke test and its report first")
    return json.loads(cfg.JUDGE_ENGINE_DECISION_PATH.read_text())


def label_path(model: str, decision: dict, limit: int | None = None) -> Path:
    gen_key = "+".join(p.stem for p in data.generation_files(model))
    key = base.content_key({
        "generations": gen_key, "stage2_extra_used": cfg.STAGE2_EXTRA_USED[model],
        "judge": cfg.JUDGE_ENGINE_WEIGHTS[decision["engine"]],
        "engine": decision["engine"], "answer_format": decision["answer_format"], "sampling": cfg.JUDGE_SAMPLING,
        "rubrics": [cfg.CALL_A_RUBRIC, cfg.CALL_A_FORMAT, cfg.CALL_B_RUBRIC, cfg.CALL_B_FORMAT, cfg.CALL_B_CUT_NOTE],
        "regex": [base.RECALL_EVAL_PATTERNS, base.RECALL_PLACEBO_PATTERNS, base.WINDOW_SENTENCES_EACH_SIDE,
                  base.WINDOW_MAX_TOKENS],
        "checks": [cfg.JUDGE_SELF_AGREEMENT_FRACTION, cfg.JUDGE_REGEX_NEGATIVE_AUDIT_FRACTION,
                   cfg.JUDGE_WINDOW_VS_FULL_PER_MODEL, cfg.JUDGE_CHECK_SEED],
    })
    return cfg.JUDGE_DIR / f"labels_{model}_{key}{f'_limit{limit}' if limit else ''}.jsonl"


def build_calls(model: str, rollouts: list[dict], texts: dict, tokenizer, answer_format: str) -> list[dict]:
    rng = random.Random(f"{cfg.JUDGE_CHECK_SEED}|{model}")
    calls, positives, negatives = [], [], []
    for r in rollouts:
        windows = cot_windows(r["cot_text"], tokenizer) if r["cot_text"] else ""
        if windows:
            positives.append(r)
            calls.append(rollout_call(r, "A", "main", call_a_message(windows, answer_format), window_text=windows))
        elif r["prompt_set"] == "fortress_adv" and r["cot_text"]:
            negatives.append(r)
        if r["status"] == "scored":
            message = call_b_message(texts[(r["prompt_set"], r["item_id"])], r["answer_text"], r["answer_cut"],
                                     answer_format)
            calls.append(rollout_call(r, "B", "main", message, answer_cut=r["answer_cut"]))
    checks = []
    for c in rng.sample(calls, math.ceil(cfg.JUDGE_SELF_AGREEMENT_FRACTION * len(calls))):
        checks.append({**c, "variant": "repeat", "seed": call_seed(c["rollout_key"], c["call"], "repeat")})
    for r in rng.sample(negatives, math.ceil(cfg.JUDGE_REGEX_NEGATIVE_AUDIT_FRACTION * len(negatives))):
        checks.append(rollout_call(r, "A", "full_cot_regex_negative", call_a_message(r["cot_text"], answer_format)))
    for r in rng.sample(positives, min(cfg.JUDGE_WINDOW_VS_FULL_PER_MODEL, len(positives))):
        checks.append(rollout_call(r, "A", "full_cot_regex_positive", call_a_message(r["cot_text"], answer_format)))
    return calls + checks


class LazyJudge:
    """The judge engine, loaded on first use, so one process can relabel a model and judge its interventions."""

    def __init__(self, engine: str):
        self.engine, self.llm, self.tokenizer = engine, None, None

    def get(self):
        if self.llm is None:
            self.llm, self.tokenizer = make_judge(self.engine), judge_tokenizer(self.engine)
        return self.llm, self.tokenizer


def run_label(models: list[str], limit: int | None, judge: LazyJudge | None = None) -> None:
    decision = judge_decision()
    todo = [m for m in models if not label_path(m, decision, limit).exists()]
    if not todo:
        print("[label] all done")
        return
    llm, tokenizer = (judge or LazyJudge(decision["engine"])).get()
    texts = data.prompt_texts()
    for model in todo:
        rollouts = data.load_rollouts(model)
        if limit:
            rollouts = rollouts[:limit]
        calls = build_calls(model, rollouts, texts, tokenizer, decision["answer_format"])
        rows, timing = run_calls(llm, tokenizer, calls, decision["answer_format"])
        write_new(label_path(model, decision, limit), rows)
        print(f"[label] {model}: {len(rows)} calls, {timing['calls_per_s']:.2f} calls/s, "
              f"parse failures {sum(not r['parse_ok'] for r in rows)}", flush=True)


# --- interventions -----------------------------------------------------------------------------------------


def run_interventions(models: list[str], smoke_bases: int | None, judge: LazyJudge | None = None) -> None:
    """Call B on every continuation answer of the insertion/removal experiment (or of its smoke run)."""
    import exp03_generate

    decision = judge_decision()
    todo = []
    for model in models:
        source = exp03_generate.intervention_path(model)
        if smoke_bases:
            source = cfg.SMOKE_DIR / f"interventions_{model}_first{smoke_bases}.jsonl"
        out = source.with_name(source.stem + "_callB.jsonl")
        if source.exists() and not out.exists():
            todo.append((model, source, out))
    if not todo:
        print("[interventions] nothing to judge")
        return
    llm, tokenizer = (judge or LazyJudge(decision["engine"])).get()
    texts = data.prompt_texts()
    for model, source, out in todo:
        calls = []
        for r in map(json.loads, source.open()):
            if r["status"] != "scored":
                continue
            key = f"{r['base_key']}|{r['experiment']}|{r['condition']}|{r['continuation']}"
            message = call_b_message(texts[(r["prompt_set"], r["item_id"])], r["answer_text"], r["answer_cut"],
                                     decision["answer_format"])
            calls.append({"model": model, "continuation_key": key, "base_key": r["base_key"],
                          "prompt_set": r["prompt_set"], "item_id": r["item_id"], "experiment": r["experiment"],
                          "condition": r["condition"], "continuation": r["continuation"], "call": "B",
                          "variant": "main", "seed": call_seed(key, "B", "intervention"), "message": message,
                          "sort_key": ("B", r["item_id"], key)})
        rows, timing = run_calls(llm, tokenizer, calls, decision["answer_format"])
        write_new(out, rows)
        print(f"[interventions] {model}: {len(rows)} calls, {timing['calls_per_s']:.2f} calls/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["smoke", "label", "interventions", "model"],
                        help="model = label, then interventions, for --models with one judge engine load")
    parser.add_argument("--engine", choices=cfg.JUDGE_ENGINES, help="smoke only")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="label only: first N rollouts per model (testing)")
    parser.add_argument("--smoke-bases", type=int, default=None, help="interventions: judge the smoke run instead")
    args = parser.parse_args()
    if args.mode == "smoke":
        if not args.engine:
            sys.exit("smoke needs --engine")
        run_smoke(args.engine)
    elif args.mode == "label":
        run_label(args.models or cfg.RELABEL_MODELS, args.limit)
    elif args.mode == "interventions":
        run_interventions(args.models or cfg.INTERVENTION_MODELS, args.smoke_bases)
    else:
        if not args.models:
            sys.exit("model mode needs --models")
        judge = LazyJudge(judge_decision()["engine"])
        run_label(args.models, None, judge)
        run_interventions([m for m in args.models if m in cfg.INTERVENTION_MODELS + cfg.INTERVENTION_MODELS_ADDED],
                          None, judge)
    # vLLM's engine teardown can abort the process after the outputs are written (it did for exp03_logit.py);
    # every output above is renamed into place when complete, so skip interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
