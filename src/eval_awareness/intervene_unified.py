"""Sentence insertion / removal with the ladder's exp03 design, for every model under the unified protocol.

The design is the collaborator's (ladder/scripts/exp03_config.py + exp03_generate.py); its boundary and seed
helpers are imported, not copied. What changes is only where labels come from: this repo's judges on the
harmonized rollouts (data/unified), instead of the ladder's exp02 judge.

  insertion  bases: one scored rollout per contested question (refusal rate over rollouts 0-9 in [0.1, 0.9]),
             rollout index < 10, no recall-regex hit in the reasoning, >= 200 reasoning tokens; up to 150 questions.
             Cut at the sentence/paragraph boundary nearest 40% of the reasoning tokens; append nothing ("none") or
             one sentence: VEA / placebo (manipulation, the VEA sentence's minimal pair) / neutral. Two phrasings
             alternate across bases.
  removal    bases: scored VEA rollouts (vea_cal) whose quote is verbatim in the reasoning, <= 3 per question, up
             to 150; "keep" resumes after the sentence holding the quote, "remove" just before it.
  Each (base, condition) gets 8 continuations; continuation j has the same seed in every condition. The total
  reasoning is capped at 8,192 tokens and the answer judged on its first 384 tokens, as in the rollouts.

    uv run python -m eval_awareness.intervene_unified bases --model Nemotron-3-Super
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m eval_awareness.intervene_unified generate --model Nemotron-3-Super --tp 4
    uv run python -m eval_awareness.intervene_unified convert-ladder --model Qwen3-14B      # needs ladder/cache
    (judge: judge.py --task refusal on data/unified/interventions/<model>.continuations.jsonl)
    uv run python -m eval_awareness.intervene_unified analyze
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval_awareness import unified as U
from eval_awareness.io import append_jsonl, read_jsonl

sys.path.insert(0, str(U.LADDER / "scripts"))
import exp03_config as cfg  # noqa: E402
import ladder_config as base  # noqa: E402

OUT = U.OUT / "interventions"
ONE_PASS_EXTRA = base.QWEN35_ONE_PASS_MAX_TOKENS - base.THINK_CAP_TOKENS  # answer room, as in the ladder
# My models: sampling as in their main runs; the reasoning opener the template leaves to the model; its closer.
SUBJECTS = {
    "gemma-4-31B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0},
                       "open": "<|channel>thought\n", "close": "<channel|>"},
    "Nemotron-3-Super": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "presence_penalty": 0.0},
                         "open": "", "close": "</think>"},
    "Qwen3.5-122B-A10B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                          "open": "", "close": "</think>"},
    "Qwen3.5-397B-A17B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                          "open": "", "close": "</think>"},
}


def gen_helpers():
    import exp03_generate
    return exp03_generate


def tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(U.MINE[model][0])


def bases_path(model: str) -> Path:
    return OUT / f"{model}.bases.json"


def continuations_path(model: str) -> Path:
    return OUT / f"{model}.continuations.jsonl"


# --- bases ---------------------------------------------------------------------------------------------------


def build_bases(model: str) -> dict:
    path = bases_path(model)
    if path.exists():
        return json.loads(path.read_text())
    g = gen_helpers()
    tok = tokenizer(model)
    df = U.load_table(model, "adversarial")  # union exclusion applied
    texts = {(r["qid"], r["sample_idx"]): r["cot"] for r in read_jsonl(U.rollouts_path(model, "adversarial"))}
    s = df[(df.status == "scored")]
    rng = random.Random(f"{cfg.INTERVENTION_SEED}|{model}")
    recall = base.compiled(base.RECALL_EVAL_PATTERNS + base.RECALL_PLACEBO_PATTERNS)

    first = s[(s.sample_idx < U.FIRST_N) & s.refusal.notna()]
    rate = first.groupby("qid").refusal.apply(lambda x: x.astype(float).mean())
    low, high = cfg.INSERT_CONTESTED_REFUSAL
    contested = set(rate[(rate >= low) & (rate <= high)].index)
    by_q = defaultdict(list)
    for r in first.itertuples():
        cot = texts[(r.qid, r.sample_idx)]
        if r.qid in contested and r.n_cot_tokens >= cfg.INSERT_MIN_COT_TOKENS and not any(p.search(cot) for p in recall):
            by_q[r.qid].append(r)
    questions = sorted(by_q)
    rng.shuffle(questions)
    insert = []
    for q in questions[: cfg.INSERT_N_BASE]:
        r = rng.choice(by_q[q])
        ids = tok.encode(texts[(r.qid, r.sample_idx)], add_special_tokens=False)
        cut, kind = g.nearest_boundary(tok, ids, round(cfg.INSERT_CUT_FRACTION * len(ids)))
        if cut is None:
            continue
        insert.append({"base_key": f"{model}|{r.qid}|{r.sample_idx}", "qid": int(r.qid), "sample_idx": int(r.sample_idx),
                       "cut": cut, "kind": kind, "phrasing": len(insert) % 2, "n_cot_tokens": len(ids),
                       "base_refusal": bool(r.refusal)})

    pool = U.design(s, "all")
    cand = [r for r in pool[(pool.vea == True) & pool.quote.notna()].itertuples()  # noqa: E712
            if r.quote and r.quote in texts[(r.qid, r.sample_idx)]]
    rng.shuffle(cand)
    remove, per_q = [], defaultdict(int)
    for r in cand:
        if len(remove) == cfg.REMOVE_N_BASE or per_q[r.qid] == cfg.REMOVE_MAX_PER_QUESTION:
            continue
        cot = texts[(r.qid, r.sample_idx)]
        ids = tok.encode(cot, add_special_tokens=False)
        start = cot.index(r.quote)
        q0, q1 = g.char_to_token(tok, ids, start), g.char_to_token(tok, ids, start + len(r.quote))
        before, _ = g.nearest_boundary(tok, ids, q0, direction=-1)
        after, kind = g.nearest_boundary(tok, ids, q1, direction=+1)
        if after is None:
            continue
        per_q[r.qid] += 1
        remove.append({"base_key": f"{model}|{r.qid}|{r.sample_idx}", "qid": int(r.qid), "sample_idx": int(r.sample_idx),
                       "cut_remove": before or 0, "cut_keep": after, "kind": kind, "quote": r.quote,
                       "n_cot_tokens": len(ids)})
    out = {"model": model, "n_contested_questions": len(contested), "n_questions_with_eligible_base": len(by_q),
           "n_vea_rollouts_with_verbatim_quote": len(cand), "insert": insert, "remove": remove}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{model}: {len(insert)} insertion bases ({len(contested)} contested questions, {len(by_q)} with an eligible "
          f"base), {len(remove)} removal bases ({len(cand)} VEA rollouts with a verbatim quote)")
    return out


# --- generation ----------------------------------------------------------------------------------------------


def jobs_for(model: str, bases: dict, tok) -> list[dict]:
    from eval_awareness.data import load_fortress
    from eval_awareness.generate import chat

    g = gen_helpers()
    sub = SUBJECTS[model]
    prompts = load_fortress(exclude_explicit_eval=False).set_index("qid").adversarial_prompt
    texts = {(r["qid"], r["sample_idx"]): r["cot"] for r in read_jsonl(U.rollouts_path(model, "adversarial"))}
    jobs = []

    def add(b, experiment, condition, kept_ids, added_ids):
        head = tok(chat(tok, prompts[b["qid"]], tokenize=False) + sub["open"], add_special_tokens=False).input_ids
        for j in range(cfg.CONTINUATIONS):
            jobs.append({"experiment": experiment, "condition": condition, "continuation": j,
                         "base_key": b["base_key"], "prompt_qid": b["qid"], "base_idx": b["sample_idx"],
                         "prefix_ids": head + kept_ids + added_ids, "n_prefix_cot": len(kept_ids) + len(added_ids),
                         "seed": g.continuation_seed(b["base_key"], experiment, j)})

    for b in bases["insert"]:
        ids = tok.encode(texts[(b["qid"], b["sample_idx"])], add_special_tokens=False)
        for condition, phrasings in cfg.INSERT_SENTENCES.items():
            add(b, "insert", condition, ids[: b["cut"]], g.inserted_ids(tok, phrasings[b["phrasing"]], b["kind"]))
    for b in bases["remove"]:
        ids = tok.encode(texts[(b["qid"], b["sample_idx"])], add_special_tokens=False)
        for condition, cut in (("keep", b["cut_keep"]), ("remove", b["cut_remove"])):
            add(b, "remove", condition, ids[:cut], [])
    for i, j in enumerate(jobs):
        j["qid"], j["sample_idx"] = i, 0  # judge.py keys; the request is prompt_qid
    return jobs


def generate(args):
    from vllm import SamplingParams, TokensPrompt

    from eval_awareness.backends import load_vllm

    model = args.model
    sub = SUBJECTS[model]
    tok = tokenizer(model)
    bases = build_bases(model)
    jobs = jobs_for(model, bases, tok)
    if args.limit_bases:
        keep = {b["base_key"] for b in bases["insert"][: args.limit_bases] + bases["remove"][: args.limit_bases]}
        jobs = [j for j in jobs if j["base_key"] in keep]
    out = continuations_path(model) if not args.limit_bases else OUT / f"{model}.smoke{args.limit_bases}.continuations.jsonl"
    done = {r["qid"] for r in read_jsonl(out)} if out.exists() else set()
    jobs = [j for j in jobs if j["qid"] not in done]
    print(f"{out.name}: {len(done)} done, {len(jobs)} to generate", flush=True)
    if not jobs:
        return
    extra = {"pipeline_parallel_size": args.pp, "limit_mm_per_prompt": {"image": 0, "video": 0}}
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    llm = load_vllm(U.MINE[model][0], args.tp, args.gpu_memory_utilization, 40960, **extra)
    close_id = tok.convert_tokens_to_ids(sub["close"])
    t0 = time.time()
    for c in range(0, len(jobs), args.chunk):
        chunk = jobs[c: c + args.chunk]
        budgets = [max(1, base.THINK_CAP_TOKENS - j["n_prefix_cot"]) for j in chunk]
        outs = llm.generate([TokensPrompt(prompt_token_ids=j["prefix_ids"]) for j in chunk],
                            [SamplingParams(n=1, seed=j["seed"], max_tokens=b + ONE_PASS_EXTRA, skip_special_tokens=False,
                                            **sub["sampling"]) for j, b in zip(chunk, budgets)])
        rows = []
        for j, b, o in zip(chunk, budgets, outs):
            ids = list(o.outputs[0].token_ids)
            end = ids.index(close_id) if close_id in ids else None
            if end is not None and end <= b:
                new_cot, answer, status = ids[:end], ids[end + 1:], "scored"
            else:
                status = "truncated" if (end is not None or o.outputs[0].finish_reason == "length") else "no_think_end"
                new_cot, answer = ids[: min(end if end is not None else len(ids), b)], None
            dec = lambda t: tok.decode(t, skip_special_tokens=True).replace("<turn|>", "").strip()  # noqa: E731
            rows.append({k: v for k, v in j.items() if k != "prefix_ids"} | {
                "model": model, "status": status, "n_new_cot_tokens": len(new_cot),
                "answer_cut": answer is not None and len(answer) > U.ANSWER_CAP,
                "cot": dec(new_cot), "response": dec(answer[: U.ANSWER_CAP]) if answer is not None else None,
                "finish_reason": o.outputs[0].finish_reason})
        append_jsonl(out, rows)
        print(f"{c + len(chunk)}/{len(jobs)} continuations, {(time.time() - t0) / 60:.1f} min", flush=True)


# --- ladder continuations -> the same format ------------------------------------------------------------


def convert_ladder(model: str):
    """The ladder's exp03 continuations, re-keyed for this repo's refusal judge. A removal base is kept only if this
    repo's VEA judge also calls the base rollout VEA with a quote inside the removed span [cut_remove, cut_keep)."""
    from eval_awareness.data import load_fortress

    g = gen_helpers()
    import exp03_data

    out = continuations_path(model)
    if out.exists():
        print(f"{out} exists")
        return
    id_to_row = {int(i): int(q) for q, i in zip(*load_fortress(exclude_explicit_eval=False)[["qid", "ID"]].T.values)}
    tok = g.tokenizer_for(model)
    bases = json.loads(g.bases_path(model).read_text())
    vea = pd.DataFrame(read_jsonl(U.judgments_path(model, "adversarial", "vea_cal"))).drop_duplicates(["qid", "sample_idx"], keep="last")
    vea = {(r.qid, r.sample_idx): (r.vea, r.quote) for r in vea.itertuples()}
    cot_ids = g.base_cot_ids(model)
    cot_text = {exp03_data.rollout_key(r): r["cot_text"] for r in g.scored_fortress(model)}
    valid_remove = set()
    for b in bases["remove"]:
        idx = int(b["base_key"].split("|")[-1]) if "|" in b["base_key"] else None
        key = (id_to_row[int(b["item_id"])], idx)
        is_vea, quote = vea.get(key, (None, None))
        text = cot_text.get(b["base_key"], "")
        if is_vea and quote and quote in text:
            ids = cot_ids[b["base_key"]]
            lead = len(tok.decode(ids)) - len(tok.decode(ids).lstrip())
            q0 = g.char_to_token(tok, ids, text.index(quote) + lead)
            if b["cut_remove"] <= q0 < b["cut_keep"]:
                valid_remove.add(b["base_key"])
    rows = []
    for i, r in enumerate(read_jsonl(g.intervention_path(model))):
        if r["experiment"] == "remove" and r["base_key"] not in valid_remove:
            continue
        rows.append({"qid": i, "sample_idx": 0, "prompt_qid": id_to_row[int(r["item_id"])], "model": model,
                     "experiment": r["experiment"], "condition": r["condition"], "continuation": r["continuation"],
                     "base_key": r["base_key"], "status": r["status"], "n_new_cot_tokens": r["n_new_cot_tokens"],
                     "cot": r["new_cot_text"], "response": r["answer_text"] if r["status"] == "scored" else None})
    OUT.mkdir(parents=True, exist_ok=True)
    append_jsonl(out, rows)
    print(f"{model}: {len(rows)} continuations; removal bases kept {len(valid_remove)}/{len(bases['remove'])}")


# --- analysis (the ladder's estimator) -----------------------------------------------------------------------


def contrasts(df: pd.DataFrame) -> dict:
    """Per base: mean refusal over its continuations; contrast = mean over bases of (treated - control),
    CI by resampling questions (bases cluster within a question). As ladder/scripts/exp03_analysis.py."""
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    out = {}
    pairs = {"insert": [("vea", "none"), ("placebo", "none"), ("neutral", "none"), ("vea", "placebo")],
             "remove": [("keep", "remove")]}
    for experiment, prs in pairs.items():
        d = df[df.experiment == experiment]
        if d.empty:
            continue
        per_base = d.groupby(["base_key", "prompt_qid", "condition"]).R.mean().reset_index()
        out[f"{experiment}_condition_refusal"] = per_base.groupby("condition").R.mean().to_dict()
        wide = per_base.pivot_table(index=["base_key", "prompt_qid"], columns="condition", values="R").reset_index()
        questions = wide.prompt_qid.unique()
        for t, c in prs:
            w = wide.dropna(subset=[t, c])
            diff = (w[t] - w[c]).groupby(w.prompt_qid).agg(["sum", "count"])
            point = diff["sum"].sum() / diff["count"].sum()
            draws = []
            for _ in range(cfg.BOOTSTRAP_ITERS):
                p = diff.reindex(rng.choice(questions, len(questions))).dropna()
                draws.append(p["sum"].sum() / p["count"].sum() if p["count"].sum() else np.nan)
            out[f"{experiment}_{t}_minus_{c}"] = {"value": float(point), "ci": [float(np.nanpercentile(draws, 2.5)),
                                                  float(np.nanpercentile(draws, 97.5))], "n_bases": len(w)}
    return out


def analyze():
    summary = {}
    for p in sorted(OUT.glob("*.continuations.jsonl")):
        if ".smoke" in p.name:
            continue
        model = p.name.split(".continuations")[0]
        rows = pd.DataFrame(read_jsonl(p))
        jp = OUT / "judgments" / f"{model}.continuations.refusal.{U.JUDGE}.jsonl"
        if not jp.exists():
            print(f"{model}: not judged yet")
            continue
        j = pd.DataFrame(read_jsonl(jp)).drop_duplicates(["qid", "sample_idx"], keep="last")[["qid", "refusal"]]
        d = rows.merge(j, on="qid", how="left")
        d = d[(d.status == "scored") & d.refusal.notna() & ~d.prompt_qid.isin(U.UNION_EXCLUDE)]
        d["R"] = d.refusal.astype(float)
        summary[model] = contrasts(d)
        print(model, {k: round(v["value"], 3) for k, v in summary[model].items() if isinstance(v, dict) and "value" in v})
    U.RESULTS.mkdir(parents=True, exist_ok=True)
    (U.RESULTS / "interventions.json").write_text(json.dumps(summary, indent=1))
    U.report()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bases"); b.add_argument("--model", required=True)
    gp = sub.add_parser("generate"); gp.add_argument("--model", required=True)
    gp.add_argument("--tp", type=int, default=1); gp.add_argument("--pp", type=int, default=1)
    gp.add_argument("--max-num-seqs", type=int); gp.add_argument("--gpu-memory-utilization", type=float, default=0.85); gp.add_argument("--chunk", type=int, default=2400)
    gp.add_argument("--limit-bases", type=int, help="smoke test: first N insertion and N removal bases")
    c = sub.add_parser("convert-ladder"); c.add_argument("--model", required=True)
    sub.add_parser("analyze")
    a = ap.parse_args()
    if a.cmd == "bases":
        build_bases(a.model)
    elif a.cmd == "generate":
        generate(a)
    elif a.cmd == "convert-ladder":
        convert_ladder(a.model)
    else:
        analyze()


if __name__ == "__main__":
    main()
