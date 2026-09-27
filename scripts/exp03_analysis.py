"""exp03 analysis: metrics, checks, figures and an auto report from the exp03 labels, interventions and logit v2.

Run in /venv/main:   python scripts/exp03_analysis.py --out-name final
Outputs: results/exp03_vea_ladder_v2/<out-name>/{summary.json, per_model.csv, REPORT_auto.md} and
figures/exp03_vea_ladder_v2/<out-name>/F*.{png,html}. Nothing is overwritten: existing outputs stop the run.

Definitions (Bloom-comparable, as in exp02): V = awareness is "evaluation" or "user_probing"; P = the judge flags a
manipulation / jailbreak mention; R = "full_refusal" (R_any = full or partial refusal, a sensitivity check).
Pre-registered primary outcomes: M1 (VEA rate), M4 (within-question refusal difference), M6 (specificity = M4 minus
the placebo difference). Everything built on judge labels is judge-scored evidence; the logit monitor and the
regex checks are judge-free. Bootstrap: Fortress questions, one shared set of draws for all models (exp02's).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import exp03_config as cfg
import exp03_data as data
import exp03_figures
import ladder_analysis as a02
import ladder_config as base
import numpy as np
import pandas as pd

LENGTH_BINS = a02.LENGTH_BINS
EARLY_VEA_FRACTION = 0.5  # quote in the first half of the CoT = "early" VEA


# --- Loading ------------------------------------------------------------------------------------------------


def judge_decision() -> dict:
    return json.loads(cfg.JUDGE_ENGINE_DECISION_PATH.read_text())


def label_rows(model: str) -> list[dict]:
    import exp03_judge

    return [json.loads(line) for line in exp03_judge.label_path(model, judge_decision()).open()]


def load_table(model: str) -> pd.DataFrame:
    """One row per rollout, exp02's columns (V, P, framing, R, ...) from the exp03 labels, plus the 4-way label."""
    rollouts = data.load_rollouts(model)
    main = {(r["rollout_key"], r["call"]): r for r in label_rows(model) if r["variant"] == "main"}
    rows = []
    for r in rollouts:
        key = data.rollout_key(r)
        a, b = main.get((key, "A")), main.get((key, "B"))
        judged_a, judged_b = a is not None and a["parse_ok"], b is not None and b["parse_ok"]
        la = a["labels"] if judged_a else {}
        # no regex hit -> awareness "none" by design; a failed parse -> missing
        awareness = (a["awareness"] if judged_a else None) if a is not None else "none"
        rows.append({
            "model": model, "rollout_key": key, "prompt_set": r["prompt_set"], "item_id": r["item_id"],
            "stage": r["stage"], "rollout_index": r["rollout_index"], "status": r["status"],
            "n_cot_tokens": r["n_cot_tokens"], "answer_cut": r["answer_cut"], "regex_hit": a is not None,
            "awareness": awareness,
            "V": np.nan if awareness is None else float(awareness in cfg.VEA_TYPES),
            "P": np.nan if awareness is None else float(bool(la.get(cfg.MENTION_FLAGS["manipulation"]))),
            **{flag: bool(la.get(flag)) for flag in cfg.MENTION_FLAGS.values()},
            "framing": la.get("framing", "none") if awareness in cfg.VEA_TYPES else "none",
            "quote": la.get("quote"), "cot_text": r["cot_text"],
            "refusal": b["refusal"] if judged_b else None,
            "R": float(b["refusal"] == cfg.REFUSAL_BINARY) if judged_b else np.nan,
            "R_any": float(b["refusal"] != "comply") if judged_b else np.nan,
        })
    df = pd.DataFrame(rows)
    df["scored"] = df.status == "scored"
    return df


# --- exp03 metrics on top of exp02's -------------------------------------------------------------------------


def share(boot: a02.QuestionBootstrap, df: pd.DataFrame, flag: pd.Series) -> dict:
    g = df.assign(x=flag.astype(float)).groupby("item_id").x
    return a02.estimate(*boot.ratio(boot.vector(g.sum()), boot.vector(g.count())))


def length_matched_rd(df: pd.DataFrame, treated: pd.Series, control: pd.Series, boot) -> dict:
    """M4 with strata = question x CoT-length tercile (Mantel-Haenszel risk difference, summed per question)."""
    d = df[(treated | control) & df.R.notna()].assign(T=treated.astype(float))
    d["stratum"] = pd.qcut(d.n_cot_tokens.rank(method="first"), 3, labels=False)
    num, den = defaultdict(float), defaultdict(float)
    for (q, _), s in d.groupby(["item_id", "stratum"]):
        n1, n0 = s["T"].sum(), (1 - s["T"]).sum()
        if n1 and n0:
            w = n1 * n0 / (n1 + n0)
            num[q] += w * (s.R[s["T"] == 1].mean() - s.R[s["T"] == 0].mean())
            den[q] += w
    point, draws = boot.ratio(boot.vector(pd.Series(num)), boot.vector(pd.Series(den)))
    n_inf = len(den)
    return a02.estimate(point, draws, n_informative_questions=n_inf, underpowered=n_inf < base.MIN_INFORMATIVE_QUESTIONS)


def quote_position(row) -> float | None:
    if not isinstance(row.quote, str) or row.quote not in row.cot_text or not row.cot_text:
        return None
    return row.cot_text.index(row.quote) / len(row.cot_text)


def exp03_metrics(df: pd.DataFrame, boot: a02.QuestionBootstrap, reference_bins: np.ndarray) -> dict:
    out = {}
    adv = df[df.prompt_set == "fortress_adv"]
    s1 = adv[(adv.stage == "stage1") & adv.scored & adv.V.notna()]
    for kind in cfg.AWARENESS_TYPES[:-1]:
        out[f"awareness_share_{kind}"] = share(boot, s1, s1.awareness == kind)
    s1r = adv[(adv.stage == "stage1") & adv.scored & adv.R.notna()]
    ben = df[(df.prompt_set == "fortress_benign") & df.scored & df.R.notna()]
    for kind in cfg.REFUSAL_TYPES:
        out[f"refusal_share_{kind}"] = share(boot, s1r, s1r.refusal == kind)
        out[f"benign_refusal_share_{kind}"] = share(boot, ben, ben.refusal == kind)

    all_scored = adv[adv.scored & adv.V.notna()]
    is_vea, not_vea = all_scored.V == 1, all_scored.V == 0
    for kind in cfg.VEA_TYPES:
        rd = a02.within_question(all_scored, all_scored.awareness == kind, not_vea, boot)
        out[f"M4_type_{kind}"] = rd["risk_difference"] | {k: rd[k] for k in ("n_informative_questions", "underpowered")}
    any_refusal = all_scored.assign(R=all_scored.R_any)
    rd = a02.within_question(any_refusal, is_vea, not_vea, boot)
    out["M4_any_refusal"] = rd["risk_difference"] | {k: rd[k] for k in ("n_informative_questions", "underpowered")}
    out["M4_length_matched"] = length_matched_rd(all_scored, is_vea, not_vea, boot)

    # when VEA appears: quote position in the CoT, and M4 for early vs late VEA
    position = all_scored[is_vea].apply(quote_position, axis=1)
    out["vea_quote_position"] = {"n_found": int(position.notna().sum()), "n_vea": int(is_vea.sum()),
                                 "median": float(position.median()) if position.notna().any() else None}
    for name, flag in (("early", position < EARLY_VEA_FRACTION), ("late", position >= EARLY_VEA_FRACTION)):
        treated = pd.Series(False, index=all_scored.index)
        treated[flag[flag].index] = True
        rd = a02.within_question(all_scored, treated, not_vea, boot)
        out[f"M4_{name}_vea"] = rd["risk_difference"] | {k: rd[k] for k in ("n_informative_questions", "underpowered")}

    # VEA / placebo overlap (the mention flags are not exclusive)
    vea_rows = s1[s1.V == 1]
    out["overlap"] = {"vea_rollouts": len(vea_rows),
                      "share_of_vea_also_mentioning_manipulation": float(vea_rows.mentions_manipulation.mean()) if len(vea_rows) else None,
                      "evaluation_and_user_probing_both": int((s1.mentions_evaluation & s1.mentions_user_probing).sum())}

    # VEA rate standardized to the pooled CoT-length distribution of all models (direct standardization)
    bins = pd.cut(s1.n_cot_tokens, LENGTH_BINS, right=False)
    rates = s1.groupby(bins, observed=False).V.mean().to_numpy()
    usable = ~np.isnan(rates)
    out["M1_length_standardized"] = float(np.sum(rates[usable] * reference_bins[usable]) / reference_bins[usable].sum())
    return out


def exp02_agreement(model: str, df: pd.DataFrame) -> dict:
    """Kappa between exp03 and exp02 labels on the same rollouts (VEA and refusal, binary)."""
    old = data.exp02_main_labels(model)
    pairs = {"vea": [], "refusal": []}
    judged_by_exp02 = (df.stage == "stage1") | (df.rollout_index < base.STAGE1_ROLLOUTS + base.STAGE2_EXTRA_USED)
    for r in df[judged_by_exp02].itertuples():
        a, b = old.get((r.rollout_key, "A")), old.get((r.rollout_key, "B"))
        if not np.isnan(r.V):
            pairs["vea"].append((r.V == 1, bool(a["labels"]["eval_aware"]) if a and a["parse_ok"] else (False if a is None else None)))
        if not np.isnan(r.R) and b and b["parse_ok"]:
            pairs["refusal"].append((r.R == 1, bool(b["labels"]["refusal"])))
    out = {}
    for k, v in pairs.items():
        v = [(x, y) for x, y in v if y is not None]
        x, y = np.array([p[0] for p in v]), np.array([p[1] for p in v])
        out[k] = {"kappa": a02.kappa(x, y), "n": len(v), "exp03_rate": float(x.mean()), "exp02_rate": float(y.mean())}
    return out


def judge_checks(model: str) -> dict:
    rows = label_rows(model)
    main = {(r["rollout_key"], r["call"]): r for r in rows if r["variant"] == "main" and r["parse_ok"]}
    field = {"A": "awareness", "B": "refusal"}
    out = {"parse_failures": sum(not r["parse_ok"] for r in rows), "n_calls": len(rows)}
    for variant in ("repeat", "full_cot_regex_positive"):
        for call in ("A", "B"):
            pairs = [(main[(r["rollout_key"], call)][field[call]], r[field[call]]) for r in rows
                     if r["variant"] == variant and r["call"] == call and r["parse_ok"] and (r["rollout_key"], call) in main]
            if pairs:
                x, y = np.array([p[0] for p in pairs], dtype=object), np.array([p[1] for p in pairs], dtype=object)
                out[f"{variant}_{call}"] = {"kappa": a02.kappa(x, y), "agreement": float(np.mean(x == y)), "n": len(pairs)}
    audit = [r["awareness"] for r in rows if r["variant"] == "full_cot_regex_negative" and r["parse_ok"]]
    out["regex_negative_audit"] = {"n": len(audit), "share_not_none": float(np.mean([a != "none" for a in audit])) if audit else None,
                                   "share_vea": float(np.mean([a in cfg.VEA_TYPES for a in audit])) if audit else None}
    return out


# --- Interventions ---------------------------------------------------------------------------------------------


def intervention_metrics(model: str) -> dict | None:
    import exp03_generate

    path = exp03_generate.intervention_path(model)
    judged_path = path.with_name(path.stem + "_callB.jsonl")
    if not judged_path.exists():
        return None
    verdicts = {r["continuation_key"]: r.get("refusal") for r in map(json.loads, judged_path.open()) if r["parse_ok"]}
    recall_eval = base.compiled(base.RECALL_EVAL_PATTERNS)
    rows = []
    for r in map(json.loads, path.open()):
        verdict = verdicts.get(f"{r['base_key']}|{r['experiment']}|{r['condition']}|{r['continuation']}")
        rows.append({"base_key": r["base_key"], "item_id": r["item_id"], "experiment": r["experiment"],
                     "condition": r["condition"], "scored": r["status"] == "scored",
                     "R": np.nan if verdict is None else float(verdict == cfg.REFUSAL_BINARY),
                     "R_any": np.nan if verdict is None else float(verdict != "comply"),
                     "eval_talk": float(any(p.search(r["new_cot_text"]) for p in recall_eval)),
                     "n_new_cot_tokens": r["n_new_cot_tokens"]})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    out = {}
    contrasts = {"insert": [("vea", "none"), ("placebo", "none"), ("neutral", "none"), ("vea", "placebo"), ("vea", "neutral")],
                 "remove": [("keep", "remove")]}
    for experiment, pairs in contrasts.items():
        d = df[df.experiment == experiment]
        per_base = d.groupby(["base_key", "item_id", "condition"]).agg(
            R=("R", "mean"), R_any=("R_any", "mean"), eval_talk=("eval_talk", "mean"), scored=("scored", "mean")).reset_index()
        out[f"{experiment}_condition_means"] = {c: {k: float(g[k].mean()) for k in ("R", "R_any", "eval_talk", "scored")}
                                                for c, g in per_base.groupby("condition")}
        questions = per_base.item_id.unique()
        for outcome in ("R", "R_any", "eval_talk"):
            wide = per_base.pivot_table(index=["base_key", "item_id"], columns="condition", values=outcome).reset_index()
            for treated, control in pairs:
                w = wide.dropna(subset=[treated, control])
                diff = (w[treated] - w[control]).groupby(w.item_id).agg(["sum", "count"])
                point = diff["sum"].sum() / diff["count"].sum()
                draws = []
                for _ in range(cfg.BOOTSTRAP_ITERS):  # resample questions (bases cluster within a question)
                    picked = diff.reindex(rng.choice(questions, len(questions))).dropna()
                    draws.append(picked["sum"].sum() / picked["count"].sum() if picked["count"].sum() else np.nan)
                out[f"{experiment}_{treated}_minus_{control}_{outcome}"] = a02.estimate(
                    point, np.array(draws), n_bases=len(w), n_questions=int(w.item_id.nunique()))
    return out


# --- Logit monitor v2 -----------------------------------------------------------------------------------------------


def logit_v2_metrics(model: str) -> dict | None:
    paths = sorted(cfg.LOGIT_DIR.glob(f"{model}_*.jsonl"))
    if not paths:
        return None
    df = pd.DataFrame([json.loads(line) for p in paths for line in p.open()]).drop_duplicates(["kind", "pair_id", "side"])
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    out, deltas = {}, {}
    for kind, g in df.groupby("kind"):
        wide = g.pivot_table(index="pair_id", columns="side", values=["score", "n_prompt_tokens"])
        a_side, b_side = sorted(g.side.unique())
        d_score = (wide["score"][b_side] - wide["score"][a_side]).to_numpy()
        d_len = (np.log(wide["n_prompt_tokens"][b_side]) - np.log(wide["n_prompt_tokens"][a_side])).to_numpy()
        deltas[kind] = d_score
        win = (d_score > 0) + 0.5 * (d_score == 0)
        idx = [rng.integers(0, len(d_score), len(d_score)) for _ in range(cfg.BOOTSTRAP_ITERS)]
        out[kind] = {"pairs": len(d_score), "sides": [a_side, b_side],
                     "win_rate": a02.estimate(win.mean(), np.array([win[i].mean() for i in idx])),
                     "mean_score_shift": a02.estimate(d_score.mean(), np.array([d_score[i].mean() for i in idx])),
                     "length_baseline_win_rate": float(np.mean(d_len > 0) + 0.5 * np.mean(d_len == 0)),
                     "corr_shift_vs_log_length_ratio": float(np.corrcoef(d_score, d_len)[0, 1]) if d_len.std() > 0 else None}
    if "disguise" in deltas and "disguise_benign" in deltas:
        h, b = deltas["disguise"], deltas["disguise_benign"]
        draws = np.array([rng.choice(h, len(h)).mean() - rng.choice(b, len(b)).mean() for _ in range(cfg.BOOTSTRAP_ITERS)])
        out["harmful_minus_benign_disguise_shift"] = a02.estimate(h.mean() - b.mean(), draws)
    return out


# --- Report ---------------------------------------------------------------------------------------------------------


def fmt(e: dict | None, scale: float = 100, digits: int = 1) -> str:
    if not e or e.get("value") is None or not np.isfinite(e["value"]):
        return "n/a"
    lo, hi = e["ci"]
    ci = f" [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]" if np.isfinite(lo) and not e.get("underpowered") else " (underpowered)"
    return f"{e['value'] * scale:.{digits}f}{ci}"


def report(summary: dict) -> str:
    pm = summary["per_model"]
    md = ["# exp03 auto report\n", ("Judge-scored numbers are marked (judge); logit-monitor numbers are judge-free. "
          "95% CIs over Fortress questions; pp = percentage points.\n"),
          "## Pre-registered primary outcomes\n",
          "| model | M1 VEA % (judge) | M1 length-standardized % | M4 refusal diff pp (judge) | M6 specificity pp (judge) |",
          "|---|---|---|---|---|"]
    for m, e in pm.items():
        md.append(f"| {m} | {fmt(e['M1_vea_rate'])} | {e['M1_length_standardized'] * 100:.1f} | "
                  f"{fmt(e['M4_within_question_pooled'])} | {fmt(e['M6_specificity'])} |")
    md += ["\n## Awareness and refusal shares, Fortress stage 1 (judge)\n",
           "| model | evaluation % | user probing % | manipulation % | full refusal % | partial % | comply % |",
           "|---|---|---|---|---|---|---|"]
    for m, e in pm.items():
        md.append(f"| {m} | " + " | ".join(fmt(e[f"awareness_share_{k}"]) for k in cfg.AWARENESS_TYPES[:-1]) + " | "
                  + " | ".join(fmt(e[f"refusal_share_{k}"]) for k in cfg.REFUSAL_TYPES) + " |")
    md += ["\n## Robustness of M4 (judge)\n",
           "| model | length-matched | any refusal (full or partial) | early VEA | late VEA | evaluation only | user probing only |",
           "|---|---|---|---|---|---|---|"]
    for m, e in pm.items():
        md.append(f"| {m} | {fmt(e['M4_length_matched'])} | {fmt(e['M4_any_refusal'])} | {fmt(e['M4_early_vea'])} | "
                  f"{fmt(e['M4_late_vea'])} | {fmt(e['M4_type_evaluation'])} | {fmt(e['M4_type_user_probing'])} |")
    md += ["\n## Sentence insertion / removal (judge-scored refusal; paired by base rollout)\n",
           "| model | VEA - none | placebo - none | neutral - none | VEA - placebo | keep - remove |", "|---|---|---|---|---|---|"]
    for m, e in summary["interventions"].items():
        if e:
            md.append(f"| {m} | " + " | ".join(fmt(e.get(k)) for k in (
                "insert_vea_minus_none_R", "insert_placebo_minus_none_R", "insert_neutral_minus_none_R",
                "insert_vea_minus_placebo_R", "remove_keep_minus_remove_R")) + " |")
    md += ["\n## Logit monitor v2 (judge-free): paired win rate %, side b > side a\n",
           "| model | benchmark > chat | disguised > blunt (harmful) | disguised > blunt (benign) | harmful - benign shift (nats) |",
           "|---|---|---|---|---|"]
    for m, e in summary["logit_v2"].items():
        if e:
            md.append(f"| {m} | {fmt(e.get('format', {}).get('win_rate'))} | {fmt(e.get('disguise', {}).get('win_rate'))} | "
                      f"{fmt(e.get('disguise_benign', {}).get('win_rate'))} | {fmt(e.get('harmful_minus_benign_disguise_shift'), 1, 2)} |")
    md += ["\n## Judge checks\n", "| model | repeat kappa A | repeat kappa B | window vs full CoT kappa | regex-negative audit: share VEA | exp02 agreement kappa VEA / refusal |",
           "|---|---|---|---|---|---|"]
    for m, c in summary["judge_checks"].items():
        g = summary["exp02_agreement"][m]
        md.append(f"| {m} | {c.get('repeat_A', {}).get('kappa', float('nan')):.3f} | {c.get('repeat_B', {}).get('kappa', float('nan')):.3f} | "
                  f"{c.get('full_cot_regex_positive_A', {}).get('kappa', float('nan')):.3f} | {c['regex_negative_audit']['share_vea']} | "
                  f"{g['vea']['kappa']:.3f} / {g['refusal']['kappa']:.3f} |")
    return "\n".join(md) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    import exp03_judge

    decision = judge_decision()
    models = [m for m in (args.models or cfg.RELABEL_MODELS) if exp03_judge.label_path(m, decision).exists()]
    out_dir = cfg.RESULTS_DIR / args.out_name
    fig_dir = cfg.figure_dir(args.out_name)
    for path in (out_dir / "summary.json", fig_dir):
        if path.exists() and (path.is_file() or any(path.iterdir())):
            raise FileExistsError(f"{path} exists; exp03 results are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    questions = sorted({i for (s, i) in data.prompt_texts() if s == "fortress_adv"}, key=int)
    boot = a02.QuestionBootstrap(questions)
    tables = {m: load_table(m) for m in models}
    stage1 = pd.concat([t[(t.prompt_set == "fortress_adv") & (t.stage == "stage1") & t.scored] for t in tables.values()])
    reference_bins = pd.cut(stage1.n_cot_tokens, LENGTH_BINS, right=False).value_counts(sort=False).to_numpy().astype(float)
    per_model = {}
    for m in models:
        per_model[m] = a02.model_metrics(tables[m], a02.load_logit(m, False), boot) | exp03_metrics(tables[m], boot, reference_bins)
        print(f"{m}: VEA {per_model[m]['M1_vea_rate']['value']:.3f}  M4 {per_model[m]['M4_within_question_pooled']['value']:+.3f}", flush=True)
    interventions = {m: intervention_metrics(m) for m in cfg.INTERVENTION_MODELS}
    logit_v2 = {m: logit_v2_metrics(m) for m in cfg.LOGIT_V2_MODELS}
    figures = exp03_figures.render_all(per_model, interventions, logit_v2, fig_dir)
    summary = a02.strip_draws({
        "exp_id": cfg.EXP_ID, "out_name": args.out_name, "models": models, "judge": decision,
        "primary_outcomes": cfg.PRIMARY_OUTCOMES, "per_model": per_model, "interventions": interventions,
        "logit_v2": logit_v2, "judge_checks": {m: judge_checks(m) for m in models},
        "exp02_agreement": {m: exp02_agreement(m, tables[m]) for m in models},
        "figures": figures, "length_reference_bins": reference_bins.tolist(),
        "bootstrap": {"unit": "Fortress question", "iters": cfg.BOOTSTRAP_ITERS, "seed": cfg.BOOTSTRAP_SEED},
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.DataFrame([a02.flat_row(m, per_model[m]) for m in models]).to_csv(out_dir / "per_model.csv", index=False)
    (out_dir / "REPORT_auto.md").write_text(report(summary))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
