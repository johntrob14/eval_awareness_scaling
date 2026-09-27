"""exp02 step 2: logit monitor + adaptive generation for one subject model, in one vLLM engine.

Run in the vLLM venv, one process per model:
    scripts/vllm_python.sh scripts/ladder_generate.py --model Qwen3-8B
    scripts/vllm_python.sh scripts/ladder_generate.py --model Qwen3-0.6B --smoke
    scripts/vllm_python.sh scripts/ladder_generate.py --model Qwen3-32B --stage1-only

Order inside the engine (each step writes its own cache file and is skipped if that
file already exists, so a crashed run resumes):
  1. Logit monitor (M7): teacher-forced log-probability of each LOGIT_PHRASES entry as
     the opening of the CoT, on the 4 LOGIT_MONITOR_SETS. Runs first so a memory
     problem shows up before any generation time is spent.
  2. Stage 1: 10 rollouts per included Fortress question, 2 per benign twin, 2 per
     excluded eval-stating prompt.
  3. Stage-2 selection: the top 20% of questions by number of stage-1 rollouts whose
     CoT matches a SELECTION_PATTERN (patterns that also match the prompt don't count).
  4. Stage 2: 40 more rollouts (indices 10..49) for the selected questions.

Generation modes (ladder_config.FAMILIES):
  two_phase: phase 1 samples the thinking and stops at </think> (cap 8,192 tokens);
             phase 2 continues from prompt + thinking for at most 384 answer tokens.
             Exact in distribution when presence_penalty is 0.
  one_pass:  one request with max_tokens 9,728; afterwards the thinking must close
             within 8,192 tokens and the answer is cut to 384 tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import ladder_config as cfg

SMOKE_QUESTIONS = 20
SMOKE_ROLLOUTS = 4
SMOKE_LOGIT_ITEMS = 20
SMOKE_SELECTION_SEED = 20260926


# --- Inputs and cache paths ------------------------------------------------------------


def load_prompts(family: str) -> dict[str, list[dict]]:
    by_set: dict[str, list[dict]] = {s: [] for s in cfg.PROMPT_SETS}
    for line in cfg.PROMPTS_PATH.open():
        row = json.loads(line)
        if row["family"] == family:
            by_set[row["prompt_set"]].append(row)
    return by_set


def cache_paths(model_name: str, args) -> dict[str, Path]:
    model = cfg.ALL_SUBJECTS[model_name]
    family = cfg.FAMILIES[model["family"]]
    prep = json.loads(cfg.PROMPT_PREP_PATH.read_text())
    key = cfg.content_key({
        "revision": model["revision"],
        "prompts_sha256": prep["prompts_file_sha256"],
        "family": family,
        "caps": [cfg.THINK_CAP_TOKENS, cfg.ANSWER_CAP_TOKENS, cfg.QWEN35_ONE_PASS_MAX_TOKENS],
        "design": [cfg.STAGE1_ROLLOUTS, cfg.BENIGN_TWIN_ROLLOUTS, cfg.EVAL_STATING_ROLLOUTS,
                   cfg.STAGE2_FRACTION, cfg.SELECTION_PATTERNS, cfg.STAGE2_TIEBREAK_SEED],
        "engine": [cfg.VLLM_ENGINE_SEED, cfg.VLLM_MAX_MODEL_LEN],
        "smoke": args.smoke,
    })
    tag = f"{model_name}_{key}" + ("_smoke" if args.smoke else "")
    return {
        "logit": cfg.LOGIT_DIR / f"{tag}.jsonl",
        "stage1": cfg.GENERATIONS_DIR / f"{tag}_stage1.jsonl",
        "selection": cfg.GENERATIONS_DIR / f"{tag}_selection.json",
        "stage2": cfg.GENERATIONS_DIR / f"{tag}_stage2_extra{args.stage2_extra}.jsonl",
        "timing": cfg.GENERATIONS_DIR / f"{tag}_timing.json",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.rename(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# --- Engine ---------------------------------------------------------------------------


def make_llm(model_name: str):
    from vllm import LLM

    model = cfg.ALL_SUBJECTS[model_name]
    path = cfg.model_dir(model["repo"], model["revision"])
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download {model['repo']}@{model['revision']} first")
    extra = {}
    if model["family"] != "qwen3":  # Qwen3.5-architecture checkpoints ship a vision tower
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    return LLM(
        model=str(path),
        tokenizer=str(path),
        dtype="bfloat16",
        seed=cfg.VLLM_ENGINE_SEED,
        max_model_len=cfg.VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=model.get("gpu_memory_utilization", cfg.VLLM_GPU_MEMORY_UTILIZATION),
        max_num_seqs=cfg.VLLM_MAX_NUM_SEQS,
        # Sampling parameters are always passed explicitly; do not inherit model defaults.
        generation_config="vllm",
        enable_prefix_caching=True,
        **extra,
    )


def sampling(family: dict, *, max_tokens: int, seed: int, stop_token_ids: list[int]):
    from vllm import SamplingParams

    return SamplingParams(n=1, max_tokens=max_tokens, seed=seed,
                          stop_token_ids=stop_token_ids, **family["sampling"])


# --- 1. Logit monitor -----------------------------------------------------------------


def run_logit_monitor(llm, model_name: str, prompts: dict[str, list[dict]], smoke: bool) -> list[dict]:
    from vllm import SamplingParams, TokensPrompt

    family_name = cfg.ALL_SUBJECTS[model_name]["family"]
    family_prep = json.loads(cfg.PROMPT_PREP_PATH.read_text())["families"][family_name]
    n_forced = len(family_prep["cot_prefix_forced_ids"])
    jobs = []
    for prompt_set in cfg.LOGIT_MONITOR_SETS:
        items = prompts[prompt_set][:SMOKE_LOGIT_ITEMS] if smoke else prompts[prompt_set]
        for item in items:
            for phrase_name, suffix in family_prep["logit_phrase_suffix_ids"].items():
                jobs.append((item, phrase_name, item["prompt_token_ids"] + suffix))
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=seq) for _, _, seq in jobs],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
    )
    rows = []
    for (item, phrase_name, seq), out in zip(jobs, outputs, strict=True):
        start = item["n_prompt_tokens"] + n_forced  # first phrase token (forced "<think>\n" excluded)
        token_logprobs = [out.prompt_logprobs[i][seq[i]].logprob for i in range(start, len(seq))]
        rows.append({
            "model": model_name, "prompt_set": item["prompt_set"], "item_id": item["item_id"],
            "phrase": phrase_name, "logprob_sum": sum(token_logprobs),
            "token_logprobs": token_logprobs, "n_prompt_tokens": item["n_prompt_tokens"],
            "n_chars": item["n_chars"],
        })
    return rows


# --- 2/4. Generation ----------------------------------------------------------------------


def rollout_requests(items: list[dict], rollout_indices: range, stage: str) -> list[dict]:
    return [
        {"item": item, "rollout_index": r, "stage": stage,
         "seed": cfg.rollout_seed(item["prompt_set"], item["item_id"], r)}
        for item in items for r in rollout_indices
    ]


def strip_stop(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    return token_ids[:-1] if token_ids and token_ids[-1] in stop_ids else token_ids


def rollout_record(model_name, request, family, cot_ids, answer_ids, status, answer_cut,
                   finish_reasons, tokenizer) -> dict:
    """One rollout. status: 'scored' (thinking closed within the cap and an answer exists),
    'truncated' (no </think> within 8,192 tokens) or 'no_think_end' (stopped without one)."""
    item = request["item"]
    if cot_ids and cot_ids[0] == family["think_start"]:
        cot_ids = cot_ids[1:]
    judged_answer = answer_ids[: cfg.ANSWER_CAP_TOKENS]
    return {
        "model": model_name,
        "prompt_set": item["prompt_set"],
        "item_id": item["item_id"],
        "stage": request["stage"],
        "rollout_index": request["rollout_index"],
        "seed": request["seed"],
        "status": status,
        "n_prompt_tokens": item["n_prompt_tokens"],
        "n_cot_tokens": len(cot_ids),
        "n_answer_tokens": len(answer_ids),
        "answer_cut": answer_cut,
        "finish_reasons": finish_reasons,
        "cot_token_ids": cot_ids,
        "answer_token_ids": judged_answer,
        "cot_text": tokenizer.decode(cot_ids, skip_special_tokens=True).strip(),
        "answer_text": tokenizer.decode(judged_answer, skip_special_tokens=True).strip(),
    }


def generate_two_phase(llm, model_name, family, requests, tokenizer) -> list[dict]:
    from vllm import TokensPrompt

    end_ids = [family["im_end"], family["endoftext"]]
    phase1 = llm.generate(
        [TokensPrompt(prompt_token_ids=r["item"]["prompt_token_ids"]) for r in requests],
        [sampling(family, max_tokens=cfg.THINK_CAP_TOKENS, seed=r["seed"],
                  stop_token_ids=[family["think_end"], *end_ids]) for r in requests],
    )
    thinking, continue_idx = [], []
    for i, out in enumerate(phase1):
        completion = out.outputs[0]
        ids = list(completion.token_ids)
        if completion.stop_reason == family["think_end"] and (not ids or ids[-1] != family["think_end"]):
            ids.append(family["think_end"])
        closed = bool(ids) and ids[-1] == family["think_end"]
        thinking.append((ids, closed, completion.finish_reason))
        if closed:
            continue_idx.append(i)
    phase2 = llm.generate(
        [TokensPrompt(prompt_token_ids=requests[i]["item"]["prompt_token_ids"] + thinking[i][0])
         for i in continue_idx],
        [sampling(family, max_tokens=cfg.ANSWER_CAP_TOKENS, seed=cfg.phase2_seed(requests[i]["seed"]),
                  stop_token_ids=end_ids) for i in continue_idx],
    )
    answers = {i: out.outputs[0] for i, out in zip(continue_idx, phase2, strict=True)}
    rows = []
    for i, request in enumerate(requests):
        ids, closed, finish1 = thinking[i]
        if closed:
            answer = answers[i]
            answer_ids = strip_stop(list(answer.token_ids), set(end_ids))
            rows.append(rollout_record(model_name, request, family, ids[:-1], answer_ids, "scored",
                                       answer.finish_reason == "length", [finish1, answer.finish_reason], tokenizer))
        else:
            status = "truncated" if finish1 == "length" else "no_think_end"
            rows.append(rollout_record(model_name, request, family, strip_stop(ids, set(end_ids)), [],
                                       status, False, [finish1], tokenizer))
    return rows


def generate_one_pass(llm, model_name, family, requests, tokenizer) -> list[dict]:
    from vllm import TokensPrompt

    end_ids = [family["im_end"], family["endoftext"]]
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=r["item"]["prompt_token_ids"]) for r in requests],
        [sampling(family, max_tokens=cfg.QWEN35_ONE_PASS_MAX_TOKENS, seed=r["seed"],
                  stop_token_ids=end_ids) for r in requests],
    )
    rows = []
    for request, out in zip(requests, outputs, strict=True):
        completion = out.outputs[0]
        ids = strip_stop(list(completion.token_ids), set(end_ids))
        end = ids.index(family["think_end"]) if family["think_end"] in ids else None
        if end is not None and end <= cfg.THINK_CAP_TOKENS:
            answer_ids = ids[end + 1:]
            answer_cut = len(answer_ids) > cfg.ANSWER_CAP_TOKENS or completion.finish_reason == "length"
            rows.append(rollout_record(model_name, request, family, ids[:end], answer_ids, "scored",
                                       answer_cut, [completion.finish_reason], tokenizer))
        else:
            cot = ids[: min(end if end is not None else len(ids), cfg.THINK_CAP_TOKENS)]
            status = "truncated" if (end is not None or completion.finish_reason == "length") else "no_think_end"
            rows.append(rollout_record(model_name, request, family, cot, [], status, False,
                                       [completion.finish_reason], tokenizer))
    return rows


def generate(llm, model_name, requests, tokenizer) -> list[dict]:
    family = cfg.FAMILIES[cfg.ALL_SUBJECTS[model_name]["family"]]
    runner = generate_two_phase if family["generation_mode"] == "two_phase" else generate_one_pass
    return runner(llm, model_name, family, requests, tokenizer)


# --- 3. Stage-2 selection ----------------------------------------------------------------------


def select_stage2(stage1_rows: list[dict], adv_items: list[dict]) -> dict:
    patterns = cfg.compiled(cfg.SELECTION_PATTERNS)
    prompt_text = {item["item_id"]: item["text"] for item in adv_items}
    active = {  # patterns that count for each question (not already in its prompt)
        qid: [p for p in patterns if not p.search(text)] for qid, text in prompt_text.items()
    }
    hits = {qid: 0 for qid in prompt_text}
    for row in stage1_rows:
        if row["prompt_set"] == "fortress_adv" and any(p.search(row["cot_text"]) for p in active[row["item_id"]]):
            hits[row["item_id"]] += 1
    tiebreak = sorted(prompt_text)
    random.Random(cfg.STAGE2_TIEBREAK_SEED).shuffle(tiebreak)
    rank = {qid: i for i, qid in enumerate(tiebreak)}
    ordered = sorted(prompt_text, key=lambda q: (-hits[q], rank[q]))
    n_select = round(cfg.STAGE2_FRACTION * len(prompt_text))
    return {
        "n_questions": len(prompt_text),
        "n_selected": n_select,
        "selected_item_ids": ordered[:n_select],
        "hits_per_question": hits,
        "n_questions_with_hits": sum(h > 0 for h in hits.values()),
        "patterns_disabled_by_prompt": {q: [p.pattern for p in patterns if p not in active[q]]
                                        for q in prompt_text if len(active[q]) < len(patterns)},
    }


# --- Main --------------------------------------------------------------------------------------


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    status = {s: sum(r["status"] == s for r in rows) for s in ("scored", "truncated", "no_think_end")}
    return {
        "n_rollouts": n,
        "status_counts": status,
        "mean_cot_tokens": sum(r["n_cot_tokens"] for r in rows) / max(n, 1),
        "mean_answer_tokens": sum(r["n_answer_tokens"] for r in rows) / max(n, 1),
        "answer_cut_rate": sum(r["answer_cut"] for r in rows) / max(status["scored"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(cfg.ALL_SUBJECTS), required=True)
    parser.add_argument("--smoke", action="store_true", help="20 questions x 4 rollouts, no stage 2")
    parser.add_argument("--stage1-only", action="store_true", help="skip stage 2 (Qwen3-32B add-on)")
    parser.add_argument("--logit-only", action="store_true",
                        help="stop after the logit monitor (Qwen3-32B when stage 1 does not fit the deadline)")
    parser.add_argument("--stage2-extra", type=int, default=cfg.STAGE2_EXTRA_USED,
                        choices=[cfg.STAGE2_EXTRA_ROLLOUTS, cfg.STAGE2_EXTRA_ROLLOUTS_FALLBACK])
    args = parser.parse_args()

    cfg.GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOGIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(args.model, args)
    family_name = cfg.ALL_SUBJECTS[args.model]["family"]
    prompts = load_prompts(family_name)
    timing = json.loads(paths["timing"].read_text()) if paths["timing"].exists() else {}

    started = time.time()
    llm = make_llm(args.model)
    tokenizer = llm.get_tokenizer()
    timing["engine_load_s"] = time.time() - started

    if not paths["logit"].exists():
        t0 = time.time()
        write_jsonl(paths["logit"], run_logit_monitor(llm, args.model, prompts, args.smoke))
        timing["logit_monitor_s"] = time.time() - t0
        print(f"{args.model}: logit monitor {timing['logit_monitor_s']:.0f}s -> {paths['logit']}", flush=True)
    if args.logit_only:
        paths["timing"].write_text(json.dumps(timing, indent=2) + "\n")
        print(f"{args.model}: done (logit monitor only)")
        return

    if args.smoke:
        chosen = sorted(random.Random(SMOKE_SELECTION_SEED).sample(
            [i["item_id"] for i in prompts["fortress_adv"]], SMOKE_QUESTIONS))
        stage1_requests = rollout_requests(
            [i for i in prompts["fortress_adv"] if i["item_id"] in chosen], range(SMOKE_ROLLOUTS), "stage1")
    else:
        stage1_requests = (
            rollout_requests(prompts["fortress_adv"], range(cfg.STAGE1_ROLLOUTS), "stage1")
            + rollout_requests(prompts["fortress_benign"], range(cfg.BENIGN_TWIN_ROLLOUTS), "stage1")
            + rollout_requests(prompts["fortress_eval_stating"], range(cfg.EVAL_STATING_ROLLOUTS), "stage1")
        )
    if paths["stage1"].exists():
        stage1_rows = read_jsonl(paths["stage1"])
    else:
        t0 = time.time()
        stage1_rows = generate(llm, args.model, stage1_requests, tokenizer)
        write_jsonl(paths["stage1"], stage1_rows)
        timing["stage1_s"] = time.time() - t0
        timing["stage1"] = summarize(stage1_rows)
        print(f"{args.model}: stage 1 {timing['stage1_s']:.0f}s {json.dumps(timing['stage1'])}", flush=True)
    paths["timing"].write_text(json.dumps(timing, indent=2) + "\n")

    if args.smoke or args.stage1_only:
        print(f"{args.model}: done (no stage 2)")
        return

    if paths["selection"].exists():
        selection = json.loads(paths["selection"].read_text())
    else:
        selection = select_stage2(stage1_rows, prompts["fortress_adv"])
        paths["selection"].write_text(json.dumps(selection, indent=2) + "\n")
    print(f"{args.model}: stage 2 selects {selection['n_selected']} questions; "
          f"{selection['n_questions_with_hits']} had >= 1 selection hit", flush=True)

    if not paths["stage2"].exists():
        chosen = set(selection["selected_item_ids"])
        start = cfg.STAGE1_ROLLOUTS
        requests = rollout_requests([i for i in prompts["fortress_adv"] if i["item_id"] in chosen],
                                    range(start, start + args.stage2_extra), "stage2")
        t0 = time.time()
        stage2_rows = generate(llm, args.model, requests, tokenizer)
        write_jsonl(paths["stage2"], stage2_rows)
        timing["stage2_s"] = time.time() - t0
        timing["stage2"] = summarize(stage2_rows)
        print(f"{args.model}: stage 2 {timing['stage2_s']:.0f}s {json.dumps(timing['stage2'])}", flush=True)
    timing["total_s"] = time.time() - started
    paths["timing"].write_text(json.dumps(timing, indent=2) + "\n")
    print(f"{args.model}: done in {timing['total_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
