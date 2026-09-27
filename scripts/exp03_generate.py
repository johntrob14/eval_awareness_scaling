"""exp03 generation: MTP smoke test, Qwen3.8 stage-2 rollouts 30..49, and the sentence insertion/removal test.

Modes (run with scripts/vllm_python.sh):
  mtp-smoke --mtp-tokens 0|2   Qwen3.8-27B on 300 Fortress questions x rollouts 0-1 (exp02 seeds), MTP off or on
  stage2                       Qwen3.8-27B rollouts 30..49 on exp02's selected questions (MTP per the decision)
  interventions --models ...   resample reasoning after inserting / removing one sentence (see below)
  bases --models ...           only build and print the intervention base rollouts and splice points (CPU)

Sentence insertion: base rollouts are exp02 stage-1 Fortress rollouts with no recall-regex hit anywhere in the
CoT, one per contested question (exp02 refusal rate 0.1-0.9). The CoT is cut at the sentence boundary nearest 40%
of its tokens; one sentence (VEA / placebo / neutral) or nothing is appended, and the model continues thinking
and answers. Removal: exp02 VEA rollouts are cut just before, or just after, the sentence containing the judge's
quote. Continuation j of a base rollout uses the same seed in every condition (paired comparison).
Caveat: vLLM applies presence penalties only to newly generated tokens, so for Qwen3.5 (penalty 1.5) the
prefix is not penalized in continuations (the same holds equally for every condition).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import exp03_config as cfg
import exp03_data as data
import ladder_config as base
import ladder_generate as gen02

# --- Engine -----------------------------------------------------------------------------------------------


def mtp_tokens_decided() -> int:
    if not cfg.MTP_DECISION_PATH.exists():
        raise FileNotFoundError(f"{cfg.MTP_DECISION_PATH} missing; run the MTP smoke test and its report first")
    return json.loads(cfg.MTP_DECISION_PATH.read_text())["mtp_tokens"]


def make_subject_llm(model_name: str, mtp_tokens: int):
    """exp02's engine settings, plus speculative decoding with the model's own MTP head if mtp_tokens > 0."""
    from vllm import LLM

    model = base.ALL_SUBJECTS[model_name]
    path = base.model_dir(model["repo"], model["revision"])
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download {model['repo']}@{model['revision']} first")
    extra = {}
    if model["family"] != "qwen3":
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if mtp_tokens > 0:
        if model["family"] not in cfg.MTP_FAMILIES:
            raise ValueError(f"{model_name} has no MTP head")
        extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": mtp_tokens}
    return LLM(model=str(path), tokenizer=str(path), dtype="bfloat16", seed=base.VLLM_ENGINE_SEED,
               max_model_len=base.VLLM_MAX_MODEL_LEN,
               gpu_memory_utilization=model.get("gpu_memory_utilization", base.VLLM_GPU_MEMORY_UTILIZATION),
               max_num_seqs=base.VLLM_MAX_NUM_SEQS, generation_config="vllm", enable_prefix_caching=True, **extra)


def mtp_for(model_name: str) -> int:
    return mtp_tokens_decided() if base.ALL_SUBJECTS[model_name]["family"] in cfg.MTP_FAMILIES else 0


def family_of(model_name: str) -> dict:
    return base.FAMILIES[base.ALL_SUBJECTS[model_name]["family"]]


def tokenizer_for(model_name: str):
    from transformers import AutoTokenizer

    model = base.ALL_SUBJECTS[model_name]
    return AutoTokenizer.from_pretrained(str(base.model_dir(model["repo"], model["revision"])))


def prompt_rows(model_name: str) -> dict[tuple[str, str], dict]:
    by_set = gen02.load_prompts(base.ALL_SUBJECTS[model_name]["family"])
    return {(row["prompt_set"], row["item_id"]): row for rows in by_set.values() for row in rows}


def write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; exp03 outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    gen02.write_jsonl(path, rows)


# --- mtp-smoke ------------------------------------------------------------------------------------------------


def mtp_smoke_path(mtp_tokens: int) -> Path:
    return cfg.SMOKE_DIR / f"mtp_{cfg.MTP_SMOKE_MODEL}_tokens{mtp_tokens}.jsonl"


def run_mtp_smoke(mtp_tokens: int) -> None:
    out = mtp_smoke_path(mtp_tokens)
    if out.exists():
        print(f"[mtp-smoke] {out.name} exists")
        return
    model = cfg.MTP_SMOKE_MODEL
    items = gen02.load_prompts(base.ALL_SUBJECTS[model]["family"])["fortress_adv"]
    items = random.Random(cfg.MTP_SMOKE_SEED).sample(sorted(items, key=lambda i: i["item_id"]), cfg.MTP_SMOKE_ITEMS)
    requests = gen02.rollout_requests(items, range(2), "stage1")  # exp02's seeds for rollouts 0 and 1
    llm = make_subject_llm(model, mtp_tokens)
    tokenizer = tokenizer_for(model)
    start = time.time()
    rows = gen02.generate(llm, model, requests, tokenizer)
    elapsed = time.time() - start
    for r in rows:
        r.pop("cot_token_ids"), r.pop("answer_token_ids")
    write_new(out, rows)
    timing = {"mtp_tokens": mtp_tokens, "n_rollouts": len(rows), "seconds": elapsed,
              "generated_tokens": sum(r["n_cot_tokens"] + r["n_answer_tokens"] for r in rows)}
    timing["tokens_per_s"] = timing["generated_tokens"] / elapsed
    out.with_name(out.stem + "_timing.json").write_text(json.dumps(timing, indent=1))
    print(f"[mtp-smoke] tokens={mtp_tokens}: {elapsed / 60:.1f} min, {timing['tokens_per_s']:.0f} tokens/s", flush=True)


# --- stage2 (Qwen3.8 rollouts 30..49) -------------------------------------------------------------------------


def run_stage2() -> None:
    model = "Qwen3.8-27B"
    out = data.qwen38_new_stage2_path()
    if out.exists():
        print(f"[stage2] {out.name} exists")
        return
    paths = gen02.cache_paths(model, SimpleNamespace(smoke=False, stage2_extra=20))
    selected = set(json.loads(paths["selection"].read_text())["selected_item_ids"])
    items = [i for i in gen02.load_prompts(base.ALL_SUBJECTS[model]["family"])["fortress_adv"]
             if i["item_id"] in selected]
    requests = gen02.rollout_requests(items, cfg.QWEN38_NEW_STAGE2, "stage2")
    mtp_tokens = mtp_tokens_decided()
    llm = make_subject_llm(model, mtp_tokens)
    start = time.time()
    rows = gen02.generate(llm, model, requests, tokenizer_for(model))
    write_new(out, rows)
    out.with_name(out.stem + "_timing.json").write_text(json.dumps(
        {"mtp_tokens": mtp_tokens, "n_rollouts": len(rows), "seconds": time.time() - start}, indent=1))
    print(f"[stage2] {model}: {len(rows)} rollouts in {(time.time() - start) / 60:.1f} min", flush=True)


# --- Sentence boundaries in token space --------------------------------------------------------------------

SENTENCE_END = re.compile(r"[.!?][\"')\]*]*$")
LIST_MARKER = re.compile(r"(?:^|\n)\s*(?:\d{1,3}|[a-zA-Z])[.)]$")  # "3." or "b)" at a line start is not a sentence


def boundary_kind(tokenizer, ids: list[int], k: int) -> str | None:
    """How ids[:k] ends: 'paragraph' (a newline), 'sentence' (. ! ? before a space), or None (not a boundary)."""
    if k <= 0 or k >= len(ids):
        return None
    before = tokenizer.decode(ids[:k])
    after = tokenizer.decode(ids[k:k + 2])
    if before.endswith("\n") and not after.startswith("\n"):
        return "paragraph"
    if SENTENCE_END.search(before) and not LIST_MARKER.search(before) and after[:1] in (" ", "\n"):
        return "sentence"
    return None


def nearest_boundary(tokenizer, ids: list[int], target: int, direction: int = 0, limit: int = 400):
    """First boundary at target, target±1, ... (direction 0), or scanning only down (-1) / up (+1)."""
    steps = [0] + [s for d in range(1, limit) for s in ((d, -d) if direction == 0 else (direction * d,))]
    for step in steps:
        kind = boundary_kind(tokenizer, ids, target + step)
        if kind:
            return target + step, kind
    return None, None


def char_to_token(tokenizer, ids: list[int], char_pos: int) -> int:
    """Smallest k such that decode(ids[:k]) covers char_pos characters (binary search on exact prefix decodes)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid])) >= char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def inserted_ids(tokenizer, sentence: str | None, kind: str) -> list[int]:
    if sentence is None:
        return []
    return tokenizer.encode(sentence if kind == "paragraph" else " " + sentence, add_special_tokens=False)


# --- Intervention bases ------------------------------------------------------------------------------------


def intervention_key(model: str) -> str:
    stems = [p.stem for p in data.exp02_generation_files(model)]
    return base.content_key({
        "generations": stems, "insert": [cfg.INSERT_N_BASE, cfg.INSERT_CONTESTED_REFUSAL, cfg.INSERT_MIN_COT_TOKENS,
                                         cfg.INSERT_CUT_FRACTION, cfg.INSERT_SENTENCES],
        "remove": [cfg.REMOVE_N_BASE, cfg.REMOVE_MAX_PER_QUESTION], "list_marker": LIST_MARKER.pattern,
        "continuations": cfg.CONTINUATIONS, "seed": cfg.INTERVENTION_SEED,
        "regex": [base.RECALL_EVAL_PATTERNS, base.RECALL_PLACEBO_PATTERNS]})


def intervention_path(model: str) -> Path:
    return cfg.INTERVENTION_DIR / f"{model}_{intervention_key(model)}.jsonl"


def bases_path(model: str) -> Path:
    return cfg.INTERVENTION_DIR / f"{model}_{intervention_key(model)}_bases.json"


def exp02_scored_fortress(model: str) -> list[dict]:
    """exp02 stage-1 and stage-2 Fortress rollouts that were scored (stage 2: the 20 extra exp02 judged)."""
    rows = []
    for path in data.exp02_generation_files(model):
        for r in map(json.loads, path.open()):
            if (r["prompt_set"] == "fortress_adv" and r["status"] == "scored"
                    and r["rollout_index"] < base.STAGE1_ROLLOUTS + base.STAGE2_EXTRA_USED):
                rows.append(r)
    return rows


def exp02_cot_ids(model: str) -> dict[str, list[int]]:
    return {data.rollout_key(r): r["cot_token_ids"] for r in exp02_scored_fortress(model)}


def build_bases(model: str, tokenizer) -> dict:
    """Base rollouts and splice points for both experiments (deterministic; cached as JSON for inspection)."""
    path = bases_path(model)
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(f"{cfg.INTERVENTION_SEED}|{model}")
    recall = base.compiled(base.RECALL_EVAL_PATTERNS + base.RECALL_PLACEBO_PATTERNS)
    labels = data.exp02_main_labels(model)
    scored = exp02_scored_fortress(model)
    stage1 = [r for r in scored if r["stage"] == "stage1"]

    refusals = defaultdict(list)
    for r in stage1:
        b = labels.get((data.rollout_key(r), "B"))
        if b and b["parse_ok"]:
            refusals[r["item_id"]].append(bool(b["labels"]["refusal"]))
    low, high = cfg.INSERT_CONTESTED_REFUSAL
    contested = {q for q, v in refusals.items() if low <= sum(v) / len(v) <= high}

    by_question = defaultdict(list)
    for r in stage1:
        if (r["item_id"] in contested and r["n_cot_tokens"] >= cfg.INSERT_MIN_COT_TOKENS
                and not any(p.search(r["cot_text"]) for p in recall)):
            by_question[r["item_id"]].append(r)
    questions = sorted(by_question)
    rng.shuffle(questions)
    insert = []
    for q in questions[: cfg.INSERT_N_BASE]:
        r = rng.choice(by_question[q])
        ids = r["cot_token_ids"]
        cut, kind = nearest_boundary(tokenizer, ids, round(cfg.INSERT_CUT_FRACTION * len(ids)))
        if cut is None:
            continue
        insert.append({"base_key": data.rollout_key(r), "prompt_set": r["prompt_set"], "item_id": q,
                       "cut": cut, "kind": kind, "phrasing": len(insert) % 2, "n_cot_tokens": len(ids),
                       "exp02_refusal": labels[(data.rollout_key(r), "B")]["labels"]["refusal"]})

    remove, per_question = [], defaultdict(int)
    candidates = [r for r in scored if (a := labels.get((data.rollout_key(r), "A"))) and a["parse_ok"]
                  and a["labels"]["eval_aware"] and a["labels"].get("quote")
                  and a["labels"]["quote"] in r["cot_text"]]
    rng.shuffle(candidates)
    for r in candidates:
        if len(remove) == cfg.REMOVE_N_BASE or per_question[r["item_id"]] == cfg.REMOVE_MAX_PER_QUESTION:
            continue
        ids, quote = r["cot_token_ids"], labels[(data.rollout_key(r), "A")]["labels"]["quote"]
        full = tokenizer.decode(ids)
        lead = len(full) - len(full.lstrip())  # cot_text is the stripped decode
        start_char = r["cot_text"].index(quote) + lead
        q_start = char_to_token(tokenizer, ids, start_char)
        q_end = char_to_token(tokenizer, ids, start_char + len(quote))
        before, _ = nearest_boundary(tokenizer, ids, q_start, direction=-1)
        after, kind = nearest_boundary(tokenizer, ids, q_end, direction=+1)
        if after is None:
            continue
        before = before or 0  # quote in the first sentence: "remove" resamples the whole CoT
        per_question[r["item_id"]] += 1
        remove.append({"base_key": data.rollout_key(r), "prompt_set": r["prompt_set"], "item_id": r["item_id"],
                       "cut_remove": before, "cut_keep": after, "kind": kind, "quote": quote,
                       "n_cot_tokens": len(ids)})
    result = {"model": model, "n_contested_questions": len(contested),
              "n_questions_with_eligible_base": len(by_question), "n_vea_rollouts_with_verbatim_quote": len(candidates),
              "insert": insert, "remove": remove}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


def continuation_seed(base_key: str, experiment: str, j: int) -> int:
    return int(hashlib.sha256(f"exp03-continue|{base_key}|{experiment}|{j}".encode()).hexdigest()[:8], 16)


def intervention_jobs(model: str, bases: dict, tokenizer) -> list[dict]:
    """One job per (base rollout, condition, continuation): the token prefix to continue from."""
    family = family_of(model)
    prompts = prompt_rows(model)
    cot_ids = exp02_cot_ids(model)
    # Qwen3's template stops before "<think>"; the model wrote it itself, and rollout_record stripped it.
    think_open = [family["think_start"]] if family["cot_prefix_forced"] else []
    jobs = []
    for b in bases["insert"]:
        prompt_ids = prompts[(b["prompt_set"], b["item_id"])]["prompt_token_ids"]
        kept = cot_ids[b["base_key"]][: b["cut"]]
        for condition, phrasings in cfg.INSERT_SENTENCES.items():
            added = inserted_ids(tokenizer, phrasings[b["phrasing"]], b["kind"])
            for j in range(cfg.CONTINUATIONS):
                jobs.append({"experiment": "insert", "condition": condition, "continuation": j, **b,
                             "prefix_ids": prompt_ids + think_open + kept + added,
                             "n_prefix_cot": len(kept) + len(added),
                             "seed": continuation_seed(b["base_key"], "insert", j)})
    for b in bases["remove"]:
        prompt_ids = prompts[(b["prompt_set"], b["item_id"])]["prompt_token_ids"]
        for condition, cut in (("keep", b["cut_keep"]), ("remove", b["cut_remove"])):
            kept = cot_ids[b["base_key"]][:cut]
            for j in range(cfg.CONTINUATIONS):
                jobs.append({"experiment": "remove", "condition": condition, "continuation": j, **b,
                             "prefix_ids": prompt_ids + think_open + kept, "n_prefix_cot": len(kept),
                             "seed": continuation_seed(b["base_key"], "remove", j)})
    return jobs


def continue_jobs(llm, model: str, jobs: list[dict], tokenizer) -> list[dict]:
    """Continue thinking from each prefix (total CoT cap 8,192 tokens), then answer (384 tokens)."""
    from vllm import TokensPrompt

    family = family_of(model)
    end_ids = [family["im_end"], family["endoftext"]]
    budget = [max(1, base.THINK_CAP_TOKENS - j["n_prefix_cot"]) for j in jobs]
    if family["generation_mode"] == "two_phase":
        phase1 = llm.generate([TokensPrompt(prompt_token_ids=j["prefix_ids"]) for j in jobs],
                              [gen02.sampling(family, max_tokens=b, seed=j["seed"],
                                              stop_token_ids=[family["think_end"], *end_ids])
                               for j, b in zip(jobs, budget)])
        thoughts = []
        for out in phase1:
            c = out.outputs[0]
            ids = list(c.token_ids)
            closed = c.stop_reason == family["think_end"] or (ids and ids[-1] == family["think_end"])
            ids = [t for t in ids if t != family["think_end"]]
            thoughts.append((gen02.strip_stop(ids, set(end_ids)), closed, c.finish_reason))
        todo = [i for i, t in enumerate(thoughts) if t[1]]
        phase2 = llm.generate([TokensPrompt(prompt_token_ids=jobs[i]["prefix_ids"] + thoughts[i][0] +
                                            [family["think_end"]]) for i in todo],
                              [gen02.sampling(family, max_tokens=base.ANSWER_CAP_TOKENS,
                                              seed=base.phase2_seed(jobs[i]["seed"]), stop_token_ids=end_ids)
                               for i in todo])
        answers = {i: out.outputs[0] for i, out in zip(todo, phase2, strict=True)}
        results = []
        for i, (new_cot, closed, finish) in enumerate(thoughts):
            if closed:
                a = answers[i]
                results.append((new_cot, gen02.strip_stop(list(a.token_ids), set(end_ids)),
                                a.finish_reason == "length", [finish, a.finish_reason], "scored"))
            else:
                status = "truncated" if finish == "length" else "no_think_end"
                results.append((new_cot, None, False, [finish], status))
    else:  # one pass (Qwen3.5): thinking and answer in one request, as in exp02
        extra = base.QWEN35_ONE_PASS_MAX_TOKENS - base.THINK_CAP_TOKENS
        outputs = llm.generate([TokensPrompt(prompt_token_ids=j["prefix_ids"]) for j in jobs],
                               [gen02.sampling(family, max_tokens=b + extra, seed=j["seed"], stop_token_ids=end_ids)
                                for j, b in zip(jobs, budget)])
        results = []
        for out, b in zip(outputs, budget):
            c = out.outputs[0]
            ids = gen02.strip_stop(list(c.token_ids), set(end_ids))
            end = ids.index(family["think_end"]) if family["think_end"] in ids else None
            if end is not None and end <= b:
                answer = ids[end + 1:]
                results.append((ids[:end], answer, len(answer) > base.ANSWER_CAP_TOKENS or c.finish_reason == "length",
                                [c.finish_reason], "scored"))
            else:
                status = "truncated" if (end is not None or c.finish_reason == "length") else "no_think_end"
                results.append((ids[: min(end if end is not None else len(ids), b)], None, False,
                                [c.finish_reason], status))
    rows = []
    for job, (new_cot, answer, cut, finishes, status) in zip(jobs, results, strict=True):
        row = {k: v for k, v in job.items() if k != "prefix_ids"}
        row.update({"model": model, "status": status,
                    "n_new_cot_tokens": len(new_cot), "answer_cut": cut, "finish_reasons": finishes,
                    "new_cot_text": tokenizer.decode(new_cot, skip_special_tokens=True),
                    "answer_text": tokenizer.decode((answer or [])[: base.ANSWER_CAP_TOKENS],
                                                    skip_special_tokens=True).strip()})
        rows.append(row)
    return rows


def show_splices(model: str, bases: dict, tokenizer, n: int = 2) -> None:
    """Print the text around a few splice points, to check that the cuts land on sentence boundaries."""
    cot_ids = exp02_cot_ids(model)
    print(f"== {model}: {len(bases['insert'])} insertion bases ({bases['n_contested_questions']} contested questions, "
          f"{bases['n_questions_with_eligible_base']} with a regex-free base), {len(bases['remove'])} removal bases")
    for b in bases["insert"][:n]:
        ids = cot_ids[b["base_key"]]
        sentence = cfg.INSERT_SENTENCES["vea"][b["phrasing"]]
        spliced = tokenizer.decode(ids[: b["cut"]] + inserted_ids(tokenizer, sentence, b["kind"]))
        print(f"-- insert ({b['kind']}, cut {b['cut']}/{len(ids)}): ...{spliced[-220:]!r}")
    for b in bases["remove"][:n]:
        ids = cot_ids[b["base_key"]]
        print(f"-- remove: quote {b['quote']!r}")
        print(f"   remove-prefix ends: ...{tokenizer.decode(ids[: b['cut_remove']])[-150:]!r}")
        print(f"   keep-prefix ends:   ...{tokenizer.decode(ids[: b['cut_keep']])[-150:]!r}")


def run_interventions(model: str, limit_bases: int | None) -> None:
    out = intervention_path(model)
    if limit_bases:
        out = cfg.SMOKE_DIR / f"interventions_{model}_first{limit_bases}.jsonl"
    if out.exists():
        print(f"[interventions] {out.name} exists")
        return
    tokenizer = tokenizer_for(model)
    bases = build_bases(model, tokenizer)
    if limit_bases:
        bases = {**bases, "insert": bases["insert"][:limit_bases], "remove": bases["remove"][:limit_bases]}
    jobs = intervention_jobs(model, bases, tokenizer)
    llm = make_subject_llm(model, mtp_for(model))
    start = time.time()
    rows = continue_jobs(llm, model, jobs, tokenizer)
    write_new(out, rows)
    out.with_name(out.stem + "_timing.json").write_text(json.dumps(
        {"mtp_tokens": mtp_for(model), "n_rollouts": len(rows), "seconds": time.time() - start}, indent=1))
    print(f"[interventions] {model}: {len(rows)} continuations in {(time.time() - start) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["mtp-smoke", "stage2", "interventions", "bases"])
    parser.add_argument("--mtp-tokens", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=cfg.INTERVENTION_MODELS)
    parser.add_argument("--limit-bases", type=int, default=None, help="interventions: first N bases (smoke)")
    args = parser.parse_args()
    if args.mode == "mtp-smoke":
        run_mtp_smoke(args.mtp_tokens)
    elif args.mode == "stage2":
        run_stage2()
    elif args.mode == "bases":
        for model in args.models:
            tokenizer = tokenizer_for(model)
            show_splices(model, build_bases(model, tokenizer), tokenizer)
    else:  # one model per process: vLLM does not reliably free GPU memory between engines
        if len(args.models) != 1:
            parser.error("interventions takes exactly one model")
        run_interventions(args.models[0], args.limit_bases)
    # vLLM's engine teardown can abort the process after the outputs are written (it did for exp03_logit.py);
    # outputs are renamed into place when complete, so skip interpreter teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
