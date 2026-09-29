"""Sentence insertion / removal with the ladder's exp03 design, for every model under the unified protocol.

The design is the collaborator's (ladder/scripts/exp03_config.py + exp03_generate.py); its boundary and seed
helpers are imported, not copied. What changes is only where labels come from: this repo's judges on the
harmonized rollouts (data/unified/<version>), instead of the ladder's exp02 judge.

  insertion  bases: one scored rollout per contested question (refusal rate over rollouts 0-9 in [0.1, 0.9]),
             rollout index < 10, no recall-regex hit in the reasoning, >= 200 reasoning tokens; up to 150 questions.
             Cut at the sentence/paragraph boundary nearest 40% of the reasoning tokens; append nothing ("none") or
             one sentence: VEA / placebo (manipulation, the VEA sentence's minimal pair) / neutral. Two phrasings
             alternate across bases.
  removal    bases: scored VEA rollouts (vea_cal) whose whole quote lies verbatim inside one removable span,
             <= 3 per question, up to 150; "keep" resumes after the sentence(s) holding the quote, "remove" just
             before them. The removed text is stored with each base for inspection.
  Each (base, condition) gets 8 continuations; continuation j has the same seed in every condition. The total
  reasoning is capped at 8,192 tokens; the refusal judge sees the first 384 answer tokens (`response`) and the
  full answer is kept (`response_full`) to check that cut on intervention outputs.

v2 changes (audits/2026-09-28/REVIEW.md): bases are built only when every scored primary rollout has valid labels;
continuation ids are a hash of (base, experiment, condition, continuation), not list positions; bases and
continuations carry manifests (sources, sampling, model snapshot, code revision) and a resume refuses a changed
base set; empty answers get their own status; the analysis keeps every assigned continuation and reports scoring
and judging rates per arm, worst-case bounds for missing outcomes, base / question counts and re-verbalization.

    uv run python -m eval_awareness.intervene_unified bases --model Nemotron-3-Super
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m eval_awareness.intervene_unified generate --model Nemotron-3-Super --tp 4
    uv run python -m eval_awareness.intervene_unified convert-ladder --model Qwen3-14B      # needs ladder/cache
    (judge: judge.py --task refusal and --task vea_cal on data/unified/<version>/interventions/<model>.continuations.jsonl)
    uv run python -m eval_awareness.intervene_unified analyze
"""

from __future__ import annotations

import argparse
import hashlib
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
PAIRS = {"insert": [("vea", "none"), ("placebo", "none"), ("neutral", "none"), ("vea", "placebo")],
         "remove": [("keep", "remove")]}


def gen_helpers():
    import exp03_generate
    return exp03_generate


def tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(U.MINE[model][0])


def model_snapshot(repo: str) -> str | None:
    """The local snapshot (revision hash) vLLM / transformers resolve for `repo`."""
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:  # noqa: BLE001
        return None


def bases_path(model: str) -> Path:
    return OUT / f"{model}.bases.json"


def continuations_path(model: str) -> Path:
    return OUT / f"{model}.continuations.jsonl"


def job_id(base_key: str, experiment: str, condition: str, j: int) -> int:
    """Stable judge key for one continuation (judge.py keys on an int qid)."""
    return int(hashlib.sha256(f"{base_key}|{experiment}|{condition}|{j}".encode()).hexdigest()[:12], 16)


# --- bases ---------------------------------------------------------------------------------------------------


def build_bases(model: str) -> dict:
    path = bases_path(model)
    rp = U.rollouts_path(model, "adversarial")
    judged = [U.judgments_path(model, "adversarial", t) for t in ("vea_cal", "refusal")]
    sources = U.source_entries([rp, *judged])
    if path.exists():
        old = json.loads(path.read_text())
        if [(x["path"], x["sha256"]) for x in old["sources"]] != [(x["path"], x["sha256"]) for x in sources]:
            raise SystemExit(f"{path.relative_to(U.ROOT)} was built from different rollouts / labels. Its continuations "
                             f"belong to those bases: keep them and bump unified.VERSION, don't rebuild in place.")
        return old
    g = gen_helpers()
    tok = tokenizer(model)
    df = U.load_table(model, "adversarial")  # union exclusion applied; invalid labels are missing
    s = df[df.status == "scored"]
    first = s[s.sample_idx < U.FIRST_N]
    pool = U.design(s, "all")
    missing = int(pool.vea.isna().sum() + pool.refusal.isna().sum())
    if missing:
        raise SystemExit(f"{model}: {missing} missing / invalid labels among scored rollouts; judge them first "
                         f"(scripts/run_unified.sh) so base selection doesn't depend on which labels happen to exist")
    texts = {(r["qid"], r["sample_idx"]): r["cot"] for r in read_jsonl(rp)}
    rng = random.Random(f"{cfg.INTERVENTION_SEED}|{model}")
    recall = base.compiled(base.RECALL_EVAL_PATTERNS + base.RECALL_PLACEBO_PATTERNS)

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
        before = before or 0
        if after is None or not (before <= q0 and q1 <= after):  # the whole quote must be inside the removed span
            continue
        removed = tok.decode(ids[before:after])
        if r.quote not in removed:
            continue
        per_q[r.qid] += 1
        remove.append({"base_key": f"{model}|{r.qid}|{r.sample_idx}", "qid": int(r.qid), "sample_idx": int(r.sample_idx),
                       "cut_remove": before, "cut_keep": after, "kind": kind, "quote": r.quote,
                       "removed_text": removed, "n_cot_tokens": len(ids)})
    out = {"model": model, "version": U.VERSION, "sources": sources, "tokenizer_snapshot": model_snapshot(U.MINE[model][0]),
           "code": U.code_revision(), "n_contested_questions": len(contested),
           "n_questions_with_eligible_base": len(by_q), "n_vea_rollouts_with_verbatim_quote": len(cand),
           "n_insert_questions": len({b["qid"] for b in insert}), "n_remove_questions": len({b["qid"] for b in remove}),
           "insert": insert, "remove": remove}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{model}: {len(insert)} insertion bases ({len(contested)} contested questions, {len(by_q)} with an eligible "
          f"base), {len(remove)} removal bases on {out['n_remove_questions']} questions ({len(cand)} VEA rollouts "
          f"with a verbatim quote)")
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
            jobs.append({"qid": job_id(b["base_key"], experiment, condition, j), "sample_idx": 0,
                         "experiment": experiment, "condition": condition, "continuation": j,
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
    man = U.manifest_path(out)
    run_config = {"bases_sha256": U.sha256_file(bases_path(model)), "sampling": sub["sampling"],
                  "think_cap": base.THINK_CAP_TOKENS, "answer_room": ONE_PASS_EXTRA, "answer_cap_judged": U.ANSWER_CAP,
                  "continuations": cfg.CONTINUATIONS, "model": U.MINE[model][0],
                  "model_snapshot": model_snapshot(U.MINE[model][0])}
    if man.exists():
        old = json.loads(man.read_text())["run_config"]
        if old != run_config:
            raise SystemExit(f"{out.relative_to(U.ROOT)} was generated with a different configuration:\n{old}\nvs now\n"
                             f"{run_config}\nBump unified.VERSION instead of mixing them in one file.")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({"version": U.VERSION, "run_config": run_config, "code": U.code_revision(),
                                   "n_jobs": len(jobs)}, indent=1))
    done = {r["qid"] for r in read_jsonl(out)}
    jobs = [j for j in jobs if j["qid"] not in done]
    print(f"{out.name}: {len(done)} done, {len(jobs)} to generate", flush=True)
    if not jobs:
        return
    extra = {"pipeline_parallel_size": args.pp, "limit_mm_per_prompt": {"image": 0, "video": 0}}
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    llm = load_vllm(U.MINE[model][0], args.tp, args.gpu_memory_utilization, 40960, **extra)
    close_id = tok.convert_tokens_to_ids(sub["close"])
    dec = lambda t: tok.decode(t, skip_special_tokens=True).replace("<turn|>", "").strip()  # noqa: E731
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
                new_cot, answer = ids[:end], ids[end + 1:]
                status = "scored" if dec(answer) else "empty_answer"
            else:
                status = "truncated" if (end is not None or o.outputs[0].finish_reason == "length") else "no_think_end"
                new_cot, answer = ids[: min(end if end is not None else len(ids), b)], None
            full = dec(answer) if answer is not None else None
            rows.append({k: v for k, v in j.items() if k != "prefix_ids"} | {
                "model": model, "status": status, "n_new_cot_tokens": len(new_cot),
                "n_answer_tokens": None if answer is None else len(answer),
                "answer_cut": answer is not None and len(answer) > U.ANSWER_CAP,
                "cot": dec(new_cot), "response": dec(answer[: U.ANSWER_CAP]) if status == "scored" else None,
                "response_full": full if status == "scored" else None, "finish_reason": o.outputs[0].finish_reason})
        append_jsonl(out, rows)
        print(f"{c + len(chunk)}/{len(jobs)} continuations, {(time.time() - t0) / 60:.1f} min", flush=True)


# --- ladder continuations -> the same format ------------------------------------------------------------


def convert_ladder(model: str):
    """The ladder's exp03 continuations, re-keyed for this repo's judges. A removal base is kept only if this repo's
    VEA judge also calls the base rollout VEA, with its whole quote inside the removed span [cut_remove, cut_keep).
    The ladder stored only the first 384 answer tokens, so there is no response_full."""
    from eval_awareness.data import load_fortress

    g = gen_helpers()
    import exp03_data

    out = continuations_path(model)
    src = g.intervention_path(model)
    bases_file = g.bases_path(model)
    vea_path = U.judgments_path(model, "adversarial", "vea_cal")
    sources = U.source_entries([src, bases_file, vea_path])
    if U.existing_state(out, sources) == "fresh":
        print(f"{out.relative_to(U.ROOT)} is up to date")
        return
    table = U.load_table(model, "adversarial")
    if table[table.status == "scored"].vea.isna().any():
        raise SystemExit(f"{model}: VEA labels incomplete; judge before converting (removal-base checks need them)")
    id_to_row = {int(i): int(q) for q, i in zip(*load_fortress(exclude_explicit_eval=False)[["qid", "ID"]].T.values)}
    tok = g.tokenizer_for(model)
    bases = json.loads(bases_file.read_text())
    vea = pd.DataFrame(read_jsonl(vea_path)).drop_duplicates(["qid", "sample_idx"], keep="last")
    vea = {(r.qid, r.sample_idx): (r.vea if getattr(r, "parse_ok", True) is not False else None, r.quote)
           for r in vea.itertuples()}
    cot_ids = g.base_cot_ids(model)
    cot_text = {exp03_data.rollout_key(r): r["cot_text"] for r in g.scored_fortress(model)}
    valid_remove, checks = set(), []
    for b in bases["remove"]:
        key = (id_to_row[int(b["item_id"])], int(b["base_key"].split("|")[-1]))
        is_vea, quote = vea.get(key, (None, None))
        text = cot_text.get(b["base_key"], "")
        ok = False
        if is_vea and quote and quote in text:
            ids = cot_ids[b["base_key"]]
            full = tok.decode(ids)
            lead = len(full) - len(full.lstrip())
            start = text.index(quote) + lead
            q0, q1 = g.char_to_token(tok, ids, start), g.char_to_token(tok, ids, start + len(quote))
            ok = b["cut_remove"] <= q0 and q1 <= b["cut_keep"]
        checks.append({"base_key": b["base_key"], "kept": ok, "vea_cal": is_vea, "vea_cal_quote": quote,
                       "removed_text": tok.decode(cot_ids[b["base_key"]][b["cut_remove"]:b["cut_keep"]])
                       if b["base_key"] in cot_ids else None})
        if ok:
            valid_remove.add(b["base_key"])
    rows = []
    for r in read_jsonl(src):
        if r["experiment"] == "remove" and r["base_key"] not in valid_remove:
            continue
        status = r["status"]
        if status == "scored" and not (r["answer_text"] or "").strip():
            status = "empty_answer"
        rows.append({"qid": job_id(r["base_key"], r["experiment"], r["condition"], r["continuation"]), "sample_idx": 0,
                     "prompt_qid": id_to_row[int(r["item_id"])], "model": model, "experiment": r["experiment"],
                     "condition": r["condition"], "continuation": r["continuation"], "base_key": r["base_key"],
                     "status": status, "n_new_cot_tokens": r["n_new_cot_tokens"], "cot": r["new_cot_text"],
                     "response": r["answer_text"] if status == "scored" else None, "response_full": None})
    U.write_with_manifest(out, rows, sources, {"removal_bases_kept": len(valid_remove),
                                               "removal_bases_total": len(bases["remove"])})
    (OUT / f"{model}.removal_checks.json").write_text(json.dumps(checks, indent=1))
    print(f"{model}: {len(rows)} continuations; removal bases kept {len(valid_remove)}/{len(bases['remove'])}")


# --- analysis (the ladder's estimator, plus missingness) ----------------------------------------------------


def contrast(per_base: pd.DataFrame, t: str, c: str, rng) -> dict:
    """Mean over bases of (treated - control), CI by resampling questions (bases cluster within a question),
    as ladder/scripts/exp03_analysis.py. per_base: one row per base, one column per condition."""
    w = per_base.dropna(subset=[t, c])
    if w.empty:
        return {"value": None, "ci": [None, None], "n_bases": 0, "n_questions": 0}
    diff = (w[t] - w[c]).groupby(w.prompt_qid).agg(["sum", "count"])
    point = diff["sum"].sum() / diff["count"].sum()
    questions = diff.index.to_numpy()
    draws = []
    for _ in range(cfg.BOOTSTRAP_ITERS):
        p = diff.loc[rng.choice(questions, len(questions))]
        draws.append(p["sum"].sum() / p["count"].sum())
    return {"value": float(point), "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_bases": len(w), "n_questions": int(w.prompt_qid.nunique())}


def contrasts(d: pd.DataFrame) -> dict:
    """d: every assigned continuation, with R (1 refusal, 0 not, NaN missing) and V (re-verbalized VEA, NaN missing).

    primary        refusal among observed outcomes (the ladder's analysis): mean over continuations per base,
                   then the paired contrast.
    bounds         worst case for missing outcomes: all missing treated = 0 and control = 1 (lower), and the
                   reverse (upper). Missing = not scored (reasoning over the cap, no answer) or not judged.
    diagnostics    per-condition counts: assigned, scored, judged; re-verbalization rate (vea_cal on the new reasoning)."""
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    out = {}
    for experiment, pairs in PAIRS.items():
        e = d[d.experiment == experiment]
        if e.empty:
            continue
        diag = e.groupby("condition").agg(assigned=("R", "size"), scored=("scored", "sum"), judged=("R", "count"),
                                          reverbalized=("V", "mean"), n_verbal_judged=("V", "count"))
        out[f"{experiment}_diagnostics"] = {c: {k: (None if v != v else float(v)) if k == "reverbalized" else int(v)
                                                for k, v in row.items()} for c, row in diag.iterrows()}
        out[f"{experiment}_n_bases"] = int(e.base_key.nunique())
        out[f"{experiment}_n_questions"] = int(e.prompt_qid.nunique())
        conds = set(e.condition)
        obs = e.groupby(["base_key", "prompt_qid", "condition"]).R.mean().unstack("condition").reset_index()
        out[f"{experiment}_condition_refusal"] = {c: float(obs[c].mean()) for c in conds if c in obs}
        for t, c in pairs:
            if t not in conds or c not in conds:
                continue
            res = {"primary": contrast(obs, t, c, rng)}
            for name, (tv, cv) in (("lower", (0.0, 1.0)), ("upper", (1.0, 0.0))):
                x = e[e.condition.isin([t, c])].copy()
                x.loc[x.R.isna() & (x.condition == t), "R"] = tv
                x.loc[x.R.isna() & (x.condition == c), "R"] = cv
                wide = x.groupby(["base_key", "prompt_qid", "condition"]).R.mean().unstack("condition").reset_index()
                res[name] = contrast(wide, t, c, rng)
            out[f"{experiment}_{t}_minus_{c}"] = res
        if experiment == "remove" and {"keep", "remove"} <= conds:
            out["remove_keep_minus_remove_rejection"] = rejection_filtered(e, rng)
    return out


REJECTION_MIN_LABELED = 0.98  # share of scored "remove" continuations that need a VEA label for the estimate


def rejection_filtered(e: pd.DataFrame, rng) -> dict:
    """Keep - remove with the post's rejection sampling: the "remove" arm keeps only continuations whose new
    reasoning does not re-verbalize VEA (vea_cal V == 0); "keep" keeps all. Bases whose every "remove" continuation
    re-verbalizes drop out. This conditions on a post-treatment outcome (continuations that avoid VEA may differ
    in other ways), which the report states next to it; it is the estimate comparable to the post's removal result."""
    rem = e[(e.condition == "remove") & e.scored]
    labeled = rem.V.notna().mean() if len(rem) else 0.0
    if labeled < REJECTION_MIN_LABELED:
        return {"primary": {"value": None, "ci": [None, None], "n_bases": 0, "n_questions": 0},
                "note": f"re-verbalization labels pending ({labeled:.0%} of removal continuations labeled)"}
    kept = e[(e.condition == "keep") | ((e.condition == "remove") & (e.V == 0))]
    wide = kept.groupby(["base_key", "prompt_qid", "condition"]).R.mean().unstack("condition").reset_index()
    res = contrast(wide, "keep", "remove", rng)
    return {"primary": res, "share_remove_kept": float((rem.V == 0).mean()),
            "bases_total": int(e.base_key.nunique()), "labeled_share": float(labeled)}


def analyze(final: bool = False):
    summary = {}
    for p in sorted(OUT.glob("*.continuations.jsonl")):
        if ".smoke" in p.name:
            continue
        model = p.name.split(".continuations")[0]
        d = pd.DataFrame(read_jsonl(p))
        d = d[~d.prompt_qid.isin(U.UNION_EXCLUDE)]
        d["scored"] = d.status == "scored"
        for task, col, name in (("refusal", "refusal", "R"), ("vea_cal", "vea", "V")):
            jp = OUT / "judgments" / f"{model}.continuations.{task}.{U.JUDGE}.jsonl"
            j = pd.DataFrame(read_jsonl(jp)) if jp.exists() else pd.DataFrame(columns=["qid", col])
            if len(j):
                j = j.drop_duplicates(["qid", "sample_idx"], keep="last")
                if "parse_ok" in j:
                    j[col] = j[col].astype(object)
                    j.loc[j.parse_ok == False, col] = None  # noqa: E712
                d = d.merge(j[["qid", col]].rename(columns={col: name}), on="qid", how="left")
            else:
                d[name] = np.nan
            d[name] = d[name].astype(float)
        d.loc[~d.scored, "R"] = np.nan  # an unscored continuation has no answer to judge
        n_unjudged = int((d.scored & d.R.isna()).sum())
        if final and n_unjudged:
            raise SystemExit(f"not final: {model} has {n_unjudged} scored continuations without a valid refusal label")
        summary[model] = contrasts(d) | {"n_assigned": len(d), "n_scored": int(d.scored.sum()),
                                         "n_scored_unjudged": n_unjudged}
        v = summary[model]
        print(model, {k: round(x["primary"]["value"], 3) for k, x in v.items()
                      if isinstance(x, dict) and "primary" in x and x["primary"]["value"] is not None})
    U.RESULTS.mkdir(parents=True, exist_ok=True)
    (U.RESULTS / "interventions.json").write_text(json.dumps(summary, indent=1))
    U.report()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bases"); b.add_argument("--model", required=True)
    gp = sub.add_parser("generate"); gp.add_argument("--model", required=True)
    gp.add_argument("--tp", type=int, default=1); gp.add_argument("--pp", type=int, default=1)
    gp.add_argument("--max-num-seqs", type=int)
    gp.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    gp.add_argument("--chunk", type=int, default=2400)
    gp.add_argument("--limit-bases", type=int, help="smoke test: first N insertion and N removal bases")
    c = sub.add_parser("convert-ladder"); c.add_argument("--model", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.cmd == "bases":
        build_bases(a.model)
    elif a.cmd == "generate":
        generate(a)
    elif a.cmd == "convert-ladder":
        convert_ladder(a.model)
    else:
        analyze(a.final)


if __name__ == "__main__":
    main()
