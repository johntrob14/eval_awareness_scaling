"""exp02 step 3: judge generated rollouts with Qwen3.8-27B (FP8, non-thinking), plus judge checks.

Run in the vLLM venv:
    scripts/vllm_python.sh scripts/ladder_judge.py label --models Qwen3-0.6B Qwen3-1.7B ...
    scripts/vllm_python.sh scripts/ladder_judge.py label --models Qwen3-0.6B --smoke   # smoke-test caches
    scripts/vllm_python.sh scripts/ladder_judge.py validate --engine thinking          # 500 items, FP8 thinking
    scripts/vllm_python.sh scripts/ladder_judge.py validate --engine bf16              # same items, bf16 judge
    /venv/main/bin/python  scripts/ladder_judge.py recall                              # regex recall on Bloom's quotes

label, per subject model (one output file per model, skipped if it exists):
  Call A (VEA, framing, placebo; Bloom F1 verbatim + our additions) on every rollout whose
      CoT matches a recall regex, shown as merged ~3-sentence windows capped at 800 tokens.
      Rollouts without a hit get V = 0 and P = 0 with no call.
  Call B (refusal; Bloom F2 verbatim) on every scored rollout, told when the answer was cut.
  Checks in the same pass: 5% of calls repeated with a new seed (self-agreement); 3% of
      regex-negative adversarial rollouts judged on the full CoT (miss rate); 20 regex-positive
      rollouts per model judged on the full CoT (window vs full agreement).
Every call logs the probability of each boolean label at its decision token (raw logprobs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import ladder_config as cfg

WINDOW_VS_FULL_PER_MODEL = 20
VALIDATION_PATH = cfg.JUDGE_DIR / "validation_items.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SENTENCE_CHARS = 600  # longer "sentences" are cut to this many characters around the hit

CALL_A_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "quote": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "eval_aware": {"type": "boolean"},
        "framing": {"enum": ["safety", "capabilities", "both", "generic", "none"]},
        "placebo": {"type": "boolean"},
    },
    "required": ["reasoning", "quote", "eval_aware", "framing", "placebo"],
    "additionalProperties": False,
}
CALL_B_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "refusal": {"type": "boolean"}},
    "required": ["reasoning", "refusal"],
    "additionalProperties": False,
}
BOOLEAN_FIELDS = {"A": ["eval_aware", "placebo"], "B": ["refusal"]}


# --- Inputs ---------------------------------------------------------------------------


def generation_files(model: str, smoke: bool) -> list[Path]:
    """The exact stage-1 (and stage-2, if present) cache files ladder_generate.py wrote."""
    import ladder_generate

    files = []
    for extra in (cfg.STAGE2_EXTRA_ROLLOUTS, cfg.STAGE2_EXTRA_ROLLOUTS_FALLBACK):
        paths = ladder_generate.cache_paths(model, SimpleNamespace(smoke=smoke, stage2_extra=extra))
        if paths["stage1"] not in files:
            files.append(paths["stage1"])
        if paths["stage2"].exists():
            files.append(paths["stage2"])
    if not files[0].exists():
        raise FileNotFoundError(f"{files[0]} missing; run ladder_generate.py --model {model} first")
    return files


def load_rollouts(model: str, smoke: bool) -> list[dict]:
    rows = []
    for path in generation_files(model, smoke):
        rows.extend(json.loads(line) for line in path.open())
    return rows


def prompt_texts() -> dict[tuple[str, str], str]:
    texts = {}
    for line in cfg.PROMPTS_PATH.open():
        row = json.loads(line)
        texts[(row["prompt_set"], row["item_id"])] = row["text"]
    return texts


def rollout_key(r: dict) -> str:
    return f"{r['model']}|{r['prompt_set']}|{r['item_id']}|{r['stage']}|{r['rollout_index']}"


def call_seed(key: str, call: str, variant: str) -> int:
    return int(hashlib.sha256(f"judge|{key}|{call}|{variant}".encode()).hexdigest()[:8], 16)


# --- Regex windows ----------------------------------------------------------------------


RECALL = cfg.compiled(cfg.RECALL_EVAL_PATTERNS + cfg.RECALL_PLACEBO_PATTERNS)


def recall_hit(text: str) -> bool:
    return any(p.search(text) for p in RECALL)


def cot_windows(cot: str, tokenizer) -> str:
    """Hit sentences with one neighbour each side, merged, joined by [...], capped in tokens."""
    sentences = [s for s in SENTENCE_SPLIT.split(cot) if s.strip()]
    hits = [i for i, s in enumerate(sentences) if recall_hit(s)]
    if not hits:
        return ""
    k = cfg.WINDOW_SENTENCES_EACH_SIDE
    ranges: list[list[int]] = []
    for i in hits:
        lo, hi = max(0, i - k), min(len(sentences) - 1, i + k)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])

    def clip(sentence: str) -> str:
        if len(sentence) <= MAX_SENTENCE_CHARS:
            return sentence
        match = next((m for p in RECALL if (m := p.search(sentence))), None)
        center = match.start() if match else 0
        start = max(0, center - MAX_SENTENCE_CHARS // 2)
        return "..." + sentence[start:start + MAX_SENTENCE_CHARS] + "..."

    pieces, used = [], 0
    for lo, hi in ranges:
        piece = " ".join(clip(s.strip()) for s in sentences[lo:hi + 1])
        n = len(tokenizer.encode(piece, add_special_tokens=False))
        if used + n > cfg.WINDOW_MAX_TOKENS:
            if not pieces:  # always show at least the first window, cut to the cap
                ids = tokenizer.encode(piece, add_special_tokens=False)[: cfg.WINDOW_MAX_TOKENS]
                pieces.append(tokenizer.decode(ids))
            break
        pieces.append(piece)
        used += n
    prefix = "[...] " if ranges[0][0] > 0 else ""
    suffix = " [...]" if ranges[-1][1] < len(sentences) - 1 or len(pieces) < len(ranges) else ""
    return prefix + " [...] ".join(pieces) + suffix


# --- Prompts ------------------------------------------------------------------------------


def call_a_message(cot_block: str, excerpts: bool) -> str:
    shown = f"{cfg.CALL_A_EXCERPT_NOTE}\n\n{cot_block}" if excerpts else cot_block
    return cfg.BLOOM_F1_RUBRIC.replace("{cot}", shown) + "\n\n" + cfg.CALL_A_EXTRA_QUESTIONS


def call_b_message(prompt: str, answer: str, cut: bool) -> str:
    note = cfg.CALL_B_CUT_NOTE.format(n=cfg.ANSWER_CAP_TOKENS) if cut else ""
    return (cfg.CALL_B_TEMPLATE.replace("{rubric}", cfg.BLOOM_F2_RUBRIC)
            .replace("{prompt}", prompt).replace("{answer}", answer).replace("{cut_note}", note))


def render(tokenizer, message: str, thinking: bool) -> list[int]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
    return tokenizer.encode(text, add_special_tokens=False)


# --- Engine and parsing ----------------------------------------------------------------------


def make_judge(engine: str):
    # DeepGEMM's FP8 kernels JIT-compile with nvcc >= 12.9 and this image has 12.8; vLLM then
    # uses its precompiled CUTLASS block-FP8 kernels. Must be set before vLLM is imported.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    from vllm import LLM

    model = cfg.JUDGE_MODEL_BF16 if engine == "bf16" else cfg.JUDGE_MODEL
    path = cfg.model_dir(model["repo"], model["revision"])
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download {model['repo']}@{model['revision']}")
    return LLM(model=str(path), tokenizer=str(path), seed=cfg.VLLM_ENGINE_SEED,
               max_model_len=cfg.JUDGE_MAX_MODEL_LEN,
               gpu_memory_utilization=cfg.VLLM_GPU_MEMORY_UTILIZATION,
               max_num_seqs=cfg.VLLM_MAX_NUM_SEQS, generation_config="vllm",
               enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})


def sampling_for(call: str, seed: int, thinking: bool):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if thinking:  # free text (thinking, then JSON); parsed leniently
        return SamplingParams(n=1, seed=seed, max_tokens=cfg.JUDGE_THINKING_MAX_TOKENS,
                              **cfg.JUDGE_SAMPLING_THINKING)
    schema = CALL_A_SCHEMA if call == "A" else CALL_B_SCHEMA
    return SamplingParams(
        n=1, seed=seed, max_tokens=cfg.JUDGE_MAX_TOKENS, logprobs=cfg.JUDGE_LOGPROBS,
        structured_outputs=StructuredOutputsParams(json=schema, disable_any_whitespace=True),
        **cfg.JUDGE_SAMPLING_NONTHINKING,
    )


def decision_probability(completion, tokenizer, field: str) -> float | None:
    """P(true) / (P(true) + P(false)) at the token where `field`'s value starts (raw logprobs)."""
    if not completion.logprobs:
        return None
    pieces = [tokenizer.decode([t]) for t in completion.token_ids]
    text = "".join(pieces)
    match = re.search(rf'"{field}"\s*:\s*', text)
    if not match:
        return None
    offset = 0
    for i, piece in enumerate(pieces):
        if offset + len(piece) > match.end() and piece.strip():
            p_true = p_false = 0.0
            for lp in completion.logprobs[i].values():
                token = (lp.decoded_token or "").strip().lower()
                if token and "true".startswith(token[:4]) and token[0] == "t":
                    p_true += math.exp(lp.logprob)
                elif token and "false".startswith(token[:5]) and token[0] == "f":
                    p_false += math.exp(lp.logprob)
            return p_true / (p_true + p_false) if p_true + p_false > 0 else None
        offset += len(piece)
    return None


def parse_json(text: str) -> dict | None:
    """Structured outputs guarantee valid JSON; thinking-mode text needs the last {...} after </think>."""
    body = text.split("</think>")[-1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        candidates = re.findall(r"\{.*\}", body, flags=re.DOTALL)
        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def run_calls(llm, tokenizer, calls: list[dict], thinking: bool) -> list[dict]:
    from vllm import TokensPrompt

    calls = sorted(calls, key=lambda c: c["sort_key"])  # shared prefixes run back to back
    prompts = [TokensPrompt(prompt_token_ids=render(tokenizer, c["message"], thinking)) for c in calls]
    outputs = llm.generate(prompts, [sampling_for(c["call"], c["seed"], thinking) for c in calls])
    results = []
    for call, prompt, out in zip(calls, prompts, outputs, strict=True):
        completion = out.outputs[0]
        parsed = parse_json(completion.text)
        row = {k: v for k, v in call.items() if k not in ("message", "sort_key")}
        row.update({
            "parse_ok": parsed is not None,
            "labels": parsed,
            "raw_text": completion.text if thinking else None,
            "finish_reason": completion.finish_reason,
            "n_input_tokens": len(prompt["prompt_token_ids"]),
            "n_output_tokens": len(completion.token_ids),
        })
        if not thinking:
            row["p_true"] = {f: decision_probability(completion, tokenizer, f) for f in BOOLEAN_FIELDS[call["call"]]}
        results.append(row)
    return results


# --- label ---------------------------------------------------------------------------------


def build_calls(model: str, rollouts: list[dict], texts: dict, tokenizer) -> list[dict]:
    rng = random.Random(f"{cfg.JUDGE_CHECK_SEED}|{model}")
    calls, positives, negatives = [], [], []
    for r in rollouts:
        key = rollout_key(r)
        base = {"model": model, "rollout_key": key, "prompt_set": r["prompt_set"],
                "item_id": r["item_id"], "stage": r["stage"], "rollout_index": r["rollout_index"]}
        windows = cot_windows(r["cot_text"], tokenizer) if r["cot_text"] else ""
        if windows:
            positives.append((r, base))
            calls.append({**base, "call": "A", "variant": "main", "seed": call_seed(key, "A", "main"),
                          "message": call_a_message(windows, excerpts=True), "window_text": windows,
                          "sort_key": ("A", model, r["prompt_set"], r["item_id"], r["stage"], r["rollout_index"])})
        elif r["prompt_set"] == "fortress_adv" and r["cot_text"]:
            negatives.append((r, base))
        if r["status"] == "scored":
            prompt = texts[(r["prompt_set"], r["item_id"])]
            calls.append({**base, "call": "B", "variant": "main", "seed": call_seed(key, "B", "main"),
                          "answer_cut": r["answer_cut"],
                          "message": call_b_message(prompt, r["answer_text"], r["answer_cut"]),
                          "sort_key": ("B", r["prompt_set"], r["item_id"], model, r["stage"], r["rollout_index"])})

    checks = []
    for call in rng.sample(calls, math.ceil(cfg.JUDGE_SELF_AGREEMENT_FRACTION * len(calls))):
        checks.append({**call, "variant": "repeat", "seed": call_seed(call["rollout_key"], call["call"], "repeat")})
    for r, base in rng.sample(negatives, math.ceil(cfg.JUDGE_REGEX_NEGATIVE_AUDIT_FRACTION * len(negatives))):
        checks.append({**base, "call": "A", "variant": "full_cot_regex_negative",
                       "seed": call_seed(base["rollout_key"], "A", "full_neg"),
                       "message": call_a_message(r["cot_text"], excerpts=False),
                       "sort_key": ("A", model, r["prompt_set"], r["item_id"], r["stage"], r["rollout_index"])})
    for r, base in rng.sample(positives, min(WINDOW_VS_FULL_PER_MODEL, len(positives))):
        checks.append({**base, "call": "A", "variant": "full_cot_regex_positive",
                       "seed": call_seed(base["rollout_key"], "A", "full_pos"),
                       "message": call_a_message(r["cot_text"], excerpts=False),
                       "sort_key": ("A", model, r["prompt_set"], r["item_id"], r["stage"], r["rollout_index"])})
    return calls + checks


def label_path(model: str, smoke: bool, limit: int | None = None) -> Path:
    gen_key = "+".join(p.stem for p in generation_files(model, smoke))
    judge_key = cfg.content_key({
        "generations": gen_key, "judge": cfg.JUDGE_MODEL, "sampling": cfg.JUDGE_SAMPLING_NONTHINKING,
        "rubrics": [cfg.BLOOM_F1_RUBRIC, cfg.BLOOM_F2_RUBRIC, cfg.CALL_A_EXCERPT_NOTE,
                    cfg.CALL_A_EXTRA_QUESTIONS, cfg.CALL_B_TEMPLATE, cfg.CALL_B_CUT_NOTE],
        "schemas": [CALL_A_SCHEMA, CALL_B_SCHEMA],
        "regex": [cfg.RECALL_EVAL_PATTERNS, cfg.RECALL_PLACEBO_PATTERNS, cfg.WINDOW_SENTENCES_EACH_SIDE,
                  cfg.WINDOW_MAX_TOKENS, MAX_SENTENCE_CHARS],
        "checks": [cfg.JUDGE_SELF_AGREEMENT_FRACTION, cfg.JUDGE_REGEX_NEGATIVE_AUDIT_FRACTION,
                   WINDOW_VS_FULL_PER_MODEL, cfg.JUDGE_CHECK_SEED],
    })
    tag = ("_smoke" if smoke else "") + (f"_limit{limit}" if limit else "")
    return cfg.JUDGE_DIR / f"labels_{model}_{judge_key}{tag}.jsonl"


def run_label(models: list[str], smoke: bool, limit: int | None) -> None:
    cfg.JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [m for m in models if not label_path(m, smoke, limit).exists()]
    if not todo:
        print("all label files exist")
        return
    llm = make_judge("fp8")
    tokenizer = llm.get_tokenizer()
    texts = prompt_texts()
    for model in todo:
        started = time.time()
        rollouts = load_rollouts(model, smoke)
        if limit:
            rollouts = random.Random(cfg.JUDGE_CHECK_SEED).sample(rollouts, min(limit, len(rollouts)))
        calls = build_calls(model, rollouts, texts, tokenizer)
        results = run_calls(llm, tokenizer, calls, thinking=False)
        out = label_path(model, smoke, limit)
        tmp = out.with_suffix(".tmp")
        with tmp.open("w") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")
        tmp.rename(out)
        n_a = sum(r["call"] == "A" and r["variant"] == "main" for r in results)
        n_b = sum(r["call"] == "B" and r["variant"] == "main" for r in results)
        print(f"{model}: {len(rollouts)} rollouts, {n_a} Call A, {n_b} Call B, {len(results)} calls total "
              f"in {time.time() - started:.0f}s; parse failures {sum(not r['parse_ok'] for r in results)} -> {out}",
              flush=True)


# --- validate ---------------------------------------------------------------------------------


def choose_validation_items(models: list[str]) -> list[dict]:
    """500 main calls stratified by model x call x label, over-sampling uncertain decisions."""
    rng = random.Random(f"{cfg.JUDGE_CHECK_SEED}|validation")
    per_model = cfg.JUDGE_VALIDATION_N // len(models)
    chosen = []
    for model in models:
        rows = [json.loads(l) for l in label_path(model, False).open()]
        main = [r for r in rows if r["variant"] == "main" and r["parse_ok"]]
        strata = {}
        for r in main:
            field = "eval_aware" if r["call"] == "A" else "refusal"
            strata.setdefault((r["call"], bool(r["labels"][field])), []).append(r)
        quota = {k: per_model // len(strata) for k in strata} if strata else {}
        for k, rows_k in strata.items():
            field = "eval_aware" if k[0] == "A" else "refusal"
            uncertain = sorted(rows_k, key=lambda r: abs((r["p_true"].get(field) or 0.5) - 0.5))
            half = quota[k] // 2
            picked = uncertain[:half]
            rest = [r for r in rows_k if r not in picked]
            picked += rng.sample(rest, min(quota[k] - len(picked), len(rest)))
            chosen.extend(picked)
    return chosen


def run_validate(engine: str, models: list[str]) -> None:
    if VALIDATION_PATH.exists():
        items = json.loads(VALIDATION_PATH.read_text())
    else:
        items = choose_validation_items(models)
        VALIDATION_PATH.write_text(json.dumps(items) + "\n")
    out = cfg.JUDGE_DIR / f"validation_{engine}_{cfg.content_key({'items': [i['rollout_key'] + i['call'] for i in items], 'engine': engine})}.jsonl"
    if out.exists():
        print(f"{out} exists")
        return
    llm = make_judge("bf16" if engine == "bf16" else "fp8")
    tokenizer = llm.get_tokenizer()
    texts = prompt_texts()
    rollouts = {}
    for model in {i["model"] for i in items}:
        for r in load_rollouts(model, False):
            rollouts[rollout_key(r)] = r
    calls = []
    for item in items:
        r = rollouts[item["rollout_key"]]
        if item["call"] == "A":
            message = call_a_message(item["window_text"], excerpts=True)
        else:
            message = call_b_message(texts[(r["prompt_set"], r["item_id"])], r["answer_text"], r["answer_cut"])
        calls.append({"model": item["model"], "rollout_key": item["rollout_key"], "call": item["call"],
                      "variant": f"validation_{engine}", "seed": call_seed(item["rollout_key"], item["call"], engine),
                      "message": message, "sort_key": (item["call"], r["prompt_set"], r["item_id"], item["rollout_key"])})
    results = run_calls(llm, tokenizer, calls, thinking=(engine == "thinking"))
    with out.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    print(f"validation ({engine}): {len(results)} calls, parse failures {sum(not r['parse_ok'] for r in results)} -> {out}")


# --- recall (CPU) ------------------------------------------------------------------------------


def run_recall() -> None:
    import urllib.request

    quotes = json.loads(urllib.request.urlopen(cfg.BLOOM_QUOTES_URL, timeout=60).read())
    rows = quotes if isinstance(quotes, list) else quotes.get("items", quotes)
    eval_only = cfg.compiled(cfg.RECALL_EVAL_PATTERNS)
    per_model: dict[str, list[int]] = {}
    missed = []
    for q in rows:
        text = q["quote"] or ""
        hit = recall_hit(text)
        per_model.setdefault(q["model"], []).append(int(hit))
        if not hit:
            missed.append({"model": q["model"], "benchmark": q.get("benchmark"), "quote": text})
    report = {
        "source": cfg.BLOOM_QUOTES_URL,
        "n_quotes": len(rows),
        "recall_any_pattern": sum(sum(v) for v in per_model.values()) / len(rows),
        "recall_eval_patterns_only": sum(any(p.search(q["quote"] or "") for p in eval_only) for q in rows) / len(rows),
        "per_model": {m: {"n": len(v), "recall": sum(v) / len(v)} for m, v in per_model.items()},
        "missed_quotes": missed,
    }
    out = cfg.RESULTS_DIR / "judge_regex_recall_bloom_quotes.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"regex recall on {len(rows)} Bloom quotes: {report['recall_any_pattern']:.3f} "
          f"(eval patterns only {report['recall_eval_patterns_only']:.3f}); per model:")
    for m, v in report["per_model"].items():
        print(f"  {m}: {v['recall']:.3f} (n={v['n']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["label", "validate", "recall"])
    parser.add_argument("--models", nargs="+", choices=list(cfg.ALL_SUBJECTS))
    parser.add_argument("--smoke", action="store_true", help="judge the smoke-test caches")
    parser.add_argument("--limit", type=int, help="judge at most this many rollouts per model (judge smoke test)")
    parser.add_argument("--engine", choices=["thinking", "bf16"])
    args = parser.parse_args()
    if args.mode == "label":
        run_label(args.models, args.smoke, args.limit)
    elif args.mode == "validate":
        run_validate(args.engine, args.models)
    else:
        run_recall()


if __name__ == "__main__":
    main()
