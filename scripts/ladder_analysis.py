"""exp02 step 4: metrics M1-M9, jump tests, figures F1-F5, summary, read samples, labeling sheets.

Run in /venv/main:
    python scripts/ladder_analysis.py --out-name preliminary_qwen3 --models Qwen3-0.6B ... Qwen3-14B
    python scripts/ladder_analysis.py --out-name final                 # every model with labels
    python scripts/ladder_analysis.py --out-name final --labeling-sheet 2

Outputs go to results/exp02_vea_ladder/<out-name>/ (final: results/exp02_vea_ladder/) and
figures/exp02_vea_ladder_<out-name>_F*.png; nothing is overwritten.

Bootstrap: every CI resamples the 489 Fortress questions with replacement, using ONE shared
set of resampled indices for all models and metrics, so differences between models (jump
tests) are computed on paired draws.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import ladder_config as cfg
import ladder_generate
import ladder_judge
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, rankdata, spearmanr

# Constant inputs (a model with no VEA on the resampled questions) give NaN draws, handled by ci().
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

FAMILY_COLORS = {"qwen3": "#1f77b4", "qwen3.5": "#ff7f0e", "qwen3.8": "#2ca02c"}
FRAMINGS = ["safety", "capabilities", "both", "generic"]
LENGTH_BINS = [0, 500, 1000, 2000, 4000, cfg.THINK_CAP_TOKENS + 1]


# --- Loading ------------------------------------------------------------------------------


def available_models(requested: list[str] | None, smoke: bool) -> list[str]:
    models = requested or [m for m in cfg.RUN_ORDER if labels_exist(m, smoke)]
    return [m for m in cfg.RUN_ORDER if m in models]


def labels_exist(model: str, smoke: bool) -> bool:
    try:
        return ladder_judge.label_path(model, smoke).exists()
    except FileNotFoundError:  # no generations yet
        return False


def load_rollout_table(model: str, smoke: bool) -> pd.DataFrame:
    """One row per rollout with the judge labels merged in (V, framing, P, R and their confidences)."""
    rollouts = ladder_judge.load_rollouts(model, smoke)
    labels = [json.loads(l) for l in ladder_judge.label_path(model, smoke).open()]
    main = {(r["rollout_key"], r["call"]): r for r in labels if r["variant"] == "main"}
    rows = []
    for r in rollouts:
        key = ladder_judge.rollout_key(r)
        a, b = main.get((key, "A")), main.get((key, "B"))
        judged_a = a is not None and a["parse_ok"]
        judged_b = b is not None and b["parse_ok"]
        rows.append({
            "model": model, "rollout_key": key, "prompt_set": r["prompt_set"], "item_id": r["item_id"],
            "stage": r["stage"], "rollout_index": r["rollout_index"], "status": r["status"],
            "n_cot_tokens": r["n_cot_tokens"], "answer_cut": r["answer_cut"],
            "regex_hit": a is not None,
            # No regex hit -> V = 0 and P = 0 by design; a failed parse -> missing.
            "V": (float(a["labels"]["eval_aware"]) if judged_a else np.nan) if a is not None else 0.0,
            "P": (float(a["labels"]["placebo"]) if judged_a else np.nan) if a is not None else 0.0,
            "framing": (a["labels"]["framing"] if judged_a else None) if a is not None else "none",
            "p_V": a["p_true"].get("eval_aware") if judged_a else None,
            "R": float(b["labels"]["refusal"]) if judged_b else np.nan,
            "p_R": b["p_true"].get("refusal") if judged_b else None,
        })
    df = pd.DataFrame(rows)
    df["scored"] = df.status == "scored"
    return df


def load_logit(model: str, smoke: bool) -> pd.DataFrame | None:
    paths = ladder_generate.cache_paths(model, SimpleNamespace(smoke=smoke, stage2_extra=cfg.STAGE2_EXTRA_ROLLOUTS))
    if not paths["logit"].exists():
        return None
    df = pd.DataFrame([json.loads(l) for l in paths["logit"].open()])
    wide = df.pivot_table(index=["prompt_set", "item_id"], columns="phrase", values="logprob_sum").reset_index()
    lengths = df.drop_duplicates(["prompt_set", "item_id"]).set_index(["prompt_set", "item_id"])["n_prompt_tokens"]
    wide["n_prompt_tokens"] = [lengths[(s, i)] for s, i in zip(wide.prompt_set, wide.item_id)]
    wide["contrast"] = wide["primary"] - wide["control"]
    return wide


# --- Per-question aggregates and the shared bootstrap ------------------------------------------


class QuestionBootstrap:
    """Shared resampling of Fortress questions: M[b, q] = times question q appears in draw b."""

    def __init__(self, question_ids: list[str]):
        self.questions = question_ids
        self.index = {q: i for i, q in enumerate(question_ids)}
        rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
        draws = rng.integers(0, len(question_ids), size=(cfg.BOOTSTRAP_ITERS, len(question_ids)))
        self.M = np.stack([np.bincount(row, minlength=len(question_ids)) for row in draws]).astype(float)
        self.draws = draws

    def vector(self, series: pd.Series) -> np.ndarray:
        """Per-question values aligned to the question order (missing questions -> 0)."""
        out = np.zeros(len(self.questions))
        for q, v in series.items():
            out[self.index[q]] = v
        return out

    def ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, np.ndarray]:
        """Point estimate and bootstrap draws of sum(num) / sum(den) over questions."""
        with np.errstate(invalid="ignore", divide="ignore"):
            point = numerator.sum() / denominator.sum() if denominator.sum() > 0 else np.nan
            draws = (self.M @ numerator) / (self.M @ denominator)
        return point, draws


def ci(draws: np.ndarray) -> list[float]:
    alpha = (1 - cfg.CI_LEVEL) / 2
    finite = draws[np.isfinite(draws)]
    if len(finite) < 0.9 * len(draws):
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))]


def estimate(point: float, draws: np.ndarray, **extra) -> dict:
    return {"value": float(point), "ci": ci(draws), "_draws": draws, **extra}


# --- M4-style within-question estimators ----------------------------------------------------------


def within_question(df: pd.DataFrame, treated: pd.Series, control: pd.Series, boot: QuestionBootstrap) -> dict:
    """Mantel-Haenszel risk difference of R between treated and control rollouts of the same question.

    w_q = n1 n0 / (n1 + n0); Delta = sum_q w_q (p1_q - p0_q) / sum_q w_q, over questions with both groups.
    Also the MH odds ratio and the question-fixed-effects linear probability model.
    """
    d = df[(treated | control) & df.R.notna()].assign(T=treated.astype(float))
    g = d.groupby("item_id")
    n1 = g.T.sum()
    n0 = g.T.count() - n1
    a = d[d["T"] == 1].groupby("item_id").R.sum().reindex(n1.index, fill_value=0)  # treated & refused
    c = d[d["T"] == 0].groupby("item_id").R.sum().reindex(n1.index, fill_value=0)  # control & refused
    both = (n1 > 0) & (n0 > 0)
    n1, n0, a, c = n1[both], n0[both], a[both], c[both]
    n = n1 + n0
    w = n1 * n0 / n
    diff = a / n1 - c / n0
    rd_point, rd_draws = boot.ratio(boot.vector(w * diff), boot.vector(w))
    b_ = n1 - a
    d_ = n0 - c
    or_point, or_draws = boot.ratio(boot.vector(a * d_ / n), boot.vector(b_ * c / n))
    # Question fixed effects: beta = sum within-question cov(T, R) / sum within-question var(T).
    dq = d[d.item_id.isin(n1.index)].copy()
    dq["Tc"] = dq["T"] - dq.groupby("item_id")["T"].transform("mean")
    dq["Rc"] = dq.R - dq.groupby("item_id").R.transform("mean")
    sxy = (dq.Tc * dq.Rc).groupby(dq.item_id).sum()
    sxx = (dq.Tc ** 2).groupby(dq.item_id).sum()
    lpm_point, lpm_draws = boot.ratio(boot.vector(sxy), boot.vector(sxx))
    n_informative = int(both.sum())
    return {
        "risk_difference": estimate(rd_point, rd_draws),
        "odds_ratio": estimate(or_point, or_draws),
        "lpm_question_fe": estimate(lpm_point, lpm_draws),
        "n_informative_questions": n_informative,
        "n_treated_rollouts_in_informative": int(n1.sum()),
        "underpowered": n_informative < cfg.MIN_INFORMATIVE_QUESTIONS,
    }


# --- Metrics for one model -------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def model_metrics(df: pd.DataFrame, logit: pd.DataFrame | None, boot: QuestionBootstrap) -> dict:
    adv = df[df.prompt_set == "fortress_adv"]
    s1 = adv[adv.stage == "stage1"]
    s1_scored = s1[s1.scored & s1.V.notna()]
    out: dict = {"n_rollouts": {f"{s}/{st}": int(n) for (s, st), n in df.groupby(["prompt_set", "stage"]).size().items()}}

    # M1: VEA rate (stage 1, scored); with truncated rollouts; pass@k.
    g = s1_scored.groupby("item_id")
    out["M1_vea_rate"] = estimate(*boot.ratio(boot.vector(g.V.sum()), boot.vector(g.V.count())),
                                  n_rollouts=int(len(s1_scored)), n_vea=int(s1_scored.V.sum()))
    s1_judged = s1[s1.V.notna() & s1.status.isin(["scored", "truncated"])]
    gj = s1_judged.groupby("item_id")
    out["M1_vea_rate_incl_truncated"] = estimate(*boot.ratio(boot.vector(gj.V.sum()), boot.vector(gj.V.count())))
    counts = g.V.agg(["count", "sum"])
    out["M1_pass_at_k"] = {k: float(np.mean([pass_at_k(int(n), int(c), k) for n, c in counts.values if n >= k]))
                           for k in range(1, cfg.STAGE1_ROLLOUTS + 1)}

    # M2: framing mix among stage-1 VEA rollouts, and per-rollout rates by framing.
    vea = s1_scored[s1_scored.V == 1]
    out["M2_framing_share"] = {f: float((vea.framing == f).mean()) if len(vea) else float("nan") for f in FRAMINGS}
    for f in FRAMINGS:
        num = boot.vector(s1_scored.assign(x=(s1_scored.V == 1) & (s1_scored.framing == f)).groupby("item_id").x.sum())
        out[f"M2_rate_{f}"] = estimate(*boot.ratio(num, boot.vector(g.V.count())))
    safety_num = boot.vector(s1_scored.assign(x=(s1_scored.V == 1) & s1_scored.framing.isin(["safety", "both"]))
                             .groupby("item_id").x.sum())
    out["M2_rate_safety_framed"] = estimate(*boot.ratio(safety_num, boot.vector(g.V.count())))

    # M3: refusal (adversarial stage 1 and benign twins).
    s1_r = s1[s1.scored & s1.R.notna()].groupby("item_id").R
    ben = df[(df.prompt_set == "fortress_benign") & df.scored & df.R.notna()].groupby("item_id").R
    adv_point, adv_draws = boot.ratio(boot.vector(s1_r.sum()), boot.vector(s1_r.count()))
    ben_point, ben_draws = boot.ratio(boot.vector(ben.sum()), boot.vector(ben.count()))
    out["M3_refusal_adversarial"] = estimate(adv_point, adv_draws)
    out["M3_refusal_benign"] = estimate(ben_point, ben_draws)
    out["M3_refusal_gap"] = estimate(adv_point - ben_point, adv_draws - ben_draws)

    # M4: within-question effect (all rollouts) and variants.
    all_scored = adv[adv.scored & adv.V.notna()]
    is_vea, not_vea = all_scored.V == 1, all_scored.V == 0
    pooled = within_question(all_scored, is_vea, not_vea, boot)
    out["M4_within_question_pooled"] = pooled["risk_difference"] | {
        k: pooled[k] for k in ("n_informative_questions", "n_treated_rollouts_in_informative", "underpowered")}
    out["M4_odds_ratio"] = pooled["odds_ratio"]
    out["M4_lpm_question_fe"] = pooled["lpm_question_fe"]
    always_refused = all_scored.groupby("item_id").R.transform("min") == 1
    kept = all_scored[~always_refused]
    bloom = within_question(kept, kept.V == 1, kept.V == 0, boot)
    out["M4_bloom_exact"] = bloom["risk_difference"] | {"n_informative_questions": bloom["n_informative_questions"]}
    s2 = all_scored[all_scored.stage == "stage2"]
    if len(s2):
        sel = within_question(s2, s2.V == 1, s2.V == 0, boot)
        out["M4_selection_free_stage2_only"] = sel["risk_difference"] | {
            k: sel[k] for k in ("n_informative_questions", "n_treated_rollouts_in_informative", "underpowered")}
    for name, framings in (("safety", ["safety", "both"]), ("capabilities", ["capabilities", "both"])):
        treated = is_vea & all_scored.framing.isin(framings)
        split = within_question(all_scored, treated, not_vea, boot)
        out[f"M4_framing_{name}"] = split["risk_difference"] | {
            k: split[k] for k in ("n_informative_questions", "n_treated_rollouts_in_informative", "underpowered")}

    # M6: placebo specificity.
    placebo = within_question(all_scored, not_vea & (all_scored.P == 1), not_vea & (all_scored.P == 0), boot)
    out["M6_placebo_within_question"] = placebo["risk_difference"] | {
        k: placebo[k] for k in ("n_informative_questions", "n_treated_rollouts_in_informative", "underpowered")}
    spec_draws = pooled["risk_difference"]["_draws"] - placebo["risk_difference"]["_draws"]
    out["M6_specificity"] = estimate(pooled["risk_difference"]["value"] - placebo["risk_difference"]["value"], spec_draws)

    # M5: across-question association (stage 1).
    per_q = s1_scored.groupby("item_id").agg(n=("V", "count"), v=("V", "mean"), n_vea=("V", "sum"), r=("R", "mean"))
    per_q["r0"] = s1_scored[s1_scored.V == 0].groupby("item_id").R.mean()
    out.update(across_question(per_q, boot))

    # M7: logit monitor.
    if logit is not None:
        out.update(logit_metrics(logit, per_q))

    # M9: hygiene.
    s1_all = s1
    out["M9_truncation_rate"] = float((s1_all.status == "truncated").mean())
    out["M9_no_think_end_rate"] = float((s1_all.status == "no_think_end").mean())
    out["M9_cot_tokens_mean"] = float(s1_all.n_cot_tokens.mean())
    out["M9_cot_tokens_median"] = float(s1_all.n_cot_tokens.median())
    out["M9_answer_cut_rate"] = float(s1_all[s1_all.scored].answer_cut.mean())
    bins = pd.cut(s1_judged.n_cot_tokens, LENGTH_BINS, right=False)
    out["M9_vea_by_cot_length"] = {str(k): {"n": int(len(v)), "vea_rate": float(v.V.mean()) if len(v) else None}
                                   for k, v in s1_judged.groupby(bins, observed=False)}

    # Positive control: the 11 excluded prompts that tell the model it is being tested.
    es = df[(df.prompt_set == "fortress_eval_stating") & df.V.notna()]
    out["positive_control_eval_stating_vea_rate"] = {"value": float(es.V.mean()) if len(es) else None, "n": int(len(es))}
    return out


def across_question(per_q: pd.DataFrame, boot: QuestionBootstrap) -> dict:
    has = per_q[per_q.r0.notna()]
    any_vea = has.n_vea > 0
    idx = np.array([boot.index[q] for q in has.index])

    def gap(values: pd.Series, flags: pd.Series, mult: np.ndarray | None) -> float:
        w = np.ones(len(values)) if mult is None else mult
        f = flags.values
        if w[f].sum() == 0 or w[~f].sum() == 0:
            return np.nan
        return np.average(values.values[f], weights=w[f]) - np.average(values.values[~f], weights=w[~f])

    mult = boot.M[:, idx]
    out = {}
    for name, col in (("M5a_gap_nonvea_refusal", "r0"), ("M5a_gap_all_refusal", "r")):
        point = gap(has[col], any_vea, None)
        draws = np.array([gap(has[col], any_vea, m) for m in mult])
        out[name] = estimate(point, draws, n_questions_with_vea=int(any_vea.sum()), n_questions=int(len(has)))
    point = spearmanr(has.v, has.r0).statistic if has.v.nunique() > 1 else np.nan
    draws = []
    for row in boot.draws:
        picked = has.reindex([boot.questions[i] for i in row]).dropna()
        draws.append(spearmanr(picked.v, picked.r0).statistic if picked.v.nunique() > 1 else np.nan)
    out["M5b_spearman_v_r0"] = estimate(point, np.array(draws))
    c6 = per_q[per_q.n_vea > 0]
    out["M5b_bloom_c6_spearman"] = {
        "value": float(spearmanr(c6.n_vea, c6.r).statistic) if len(c6) > 2 and c6.n_vea.nunique() > 1 else None,
        "n_questions": int(len(c6)),
    }
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > random negative), ties count 1/2 (Mann-Whitney)."""
    ranks = rankdata(np.concatenate([pos, neg]))
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def logit_metrics(logit: pd.DataFrame, per_q: pd.DataFrame) -> dict:
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    out = {}
    scores = ["primary", "contrast", *[p for p in cfg.LOGIT_PHRASES if p.startswith("alt_")]]
    pos = logit[logit.prompt_set == "fortress_adv"]
    for other in cfg.LOGIT_MONITOR_CONTRASTS:
        neg = logit[logit.prompt_set == other]
        if not len(neg):
            continue
        both = pd.concat([pos, neg])
        log_len = np.log(both.n_prompt_tokens.values)
        is_pos = (both.prompt_set == "fortress_adv").values
        columns = {s: both[s].values for s in scores}
        slope, intercept = np.polyfit(log_len, both["primary"].values, 1)
        columns["primary_length_residualized"] = both["primary"].values - (slope * log_len + intercept)
        columns["length_only_baseline"] = log_len
        entry = {}
        for name, values in columns.items():
            p_vals, n_vals = values[is_pos], values[~is_pos]
            point = auroc(p_vals, n_vals)
            draws = np.array([auroc(rng.choice(p_vals, len(p_vals)), rng.choice(n_vals, len(n_vals)))
                              for _ in range(cfg.BOOTSTRAP_ITERS)])
            entry[name] = estimate(point, draws)
        out[f"M7_auroc_vs_{other}"] = entry
    s_q = pos.set_index("item_id")["primary"]
    joined = per_q.join(s_q.rename("s"), how="inner")
    out["M7b_spearman_s_v"] = {"value": float(spearmanr(joined.s, joined.v).statistic) if joined.v.nunique() > 1 else None,
                               "n_questions": int(len(joined))}
    return out


# --- Judge checks (M8) ------------------------------------------------------------------------------------


def kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0:
        return None
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else None


def judge_checks(models: list[str], smoke: bool) -> dict:
    out = {"per_model": {}}
    for model in models:
        rows = [json.loads(l) for l in ladder_judge.label_path(model, smoke).open()]
        main = {(r["rollout_key"], r["call"]): r for r in rows if r["variant"] == "main" and r["parse_ok"]}
        entry = {"parse_failures": int(sum(not r["parse_ok"] for r in rows)), "n_calls": len(rows)}
        for call, field in (("A", "eval_aware"), ("B", "refusal"), ("A", "placebo")):
            pairs = [(main[(r["rollout_key"], call)]["labels"][field], r["labels"][field]) for r in rows
                     if r["variant"] == "repeat" and r["call"] == call and r["parse_ok"] and (r["rollout_key"], call) in main]
            arr = np.array(pairs, dtype=float).reshape(-1, 2)
            entry[f"self_agreement_{field}"] = {"n": len(arr), "agreement": float(np.mean(arr[:, 0] == arr[:, 1])) if len(arr) else None,
                                                "kappa": kappa(arr[:, 0], arr[:, 1]) if len(arr) else None}
        neg = [r for r in rows if r["variant"] == "full_cot_regex_negative" and r["parse_ok"]]
        entry["regex_negative_audit"] = {"n": len(neg), "vea_rate_full_cot": float(np.mean([r["labels"]["eval_aware"] for r in neg])) if neg else None}
        pos = [(main[(r["rollout_key"], "A")]["labels"]["eval_aware"], r["labels"]["eval_aware"]) for r in rows
               if r["variant"] == "full_cot_regex_positive" and r["parse_ok"] and (r["rollout_key"], "A") in main]
        arr = np.array(pos, dtype=float).reshape(-1, 2)
        entry["window_vs_full_cot"] = {"n": len(arr), "agreement": float(np.mean(arr[:, 0] == arr[:, 1])) if len(arr) else None,
                                       "kappa": kappa(arr[:, 0], arr[:, 1]) if len(arr) else None}
        out["per_model"][model] = entry
    for engine in ("thinking", "bf16"):
        files = sorted(cfg.JUDGE_DIR.glob(f"validation_{engine}_*.jsonl"))
        if not files or smoke:
            continue
        val = [json.loads(l) for l in files[-1].open()]
        mains = {}
        for model in {v["model"] for v in val}:
            for r in map(json.loads, ladder_judge.label_path(model, False).open()):
                if r["variant"] == "main" and r["parse_ok"]:
                    mains[(r["rollout_key"], r["call"])] = r
        res = {"parse_failures": int(sum(not v["parse_ok"] for v in val)), "n": len(val)}
        for call, field in (("A", "eval_aware"), ("B", "refusal")):
            pairs = [(mains[(v["rollout_key"], call)]["labels"][field], bool(v["labels"].get(field)))
                     for v in val if v["call"] == call and v["parse_ok"] and isinstance(v["labels"], dict)
                     and (v["rollout_key"], call) in mains]
            arr = np.array(pairs, dtype=float).reshape(-1, 2)
            res[field] = {"n": len(arr), "agreement": float(np.mean(arr[:, 0] == arr[:, 1])) if len(arr) else None,
                          "kappa": kappa(arr[:, 0], arr[:, 1]) if len(arr) else None,
                          "note": "stratified sample over-weighting uncertain items; not a population agreement rate"}
        out[f"validation_{engine}_vs_main"] = res
    recall = cfg.RESULTS_DIR / "judge_regex_recall_bloom_quotes.json"
    if recall.exists():
        r = json.loads(recall.read_text())
        out["regex_recall_bloom_quotes"] = {k: r[k] for k in ("n_quotes", "recall_any_pattern", "recall_eval_patterns_only", "per_model")}
    return out


# --- Jump tests -------------------------------------------------------------------------------------------


def jump_tests(metrics: dict[str, dict], models: list[str]) -> dict:
    out = {}
    for outcome in cfg.PRIMARY_OUTCOMES:
        for family in ("qwen3", "qwen3.5"):
            ladder = [m for m in models if cfg.ALL_SUBJECTS[m]["family"] == family and outcome in metrics[m]
                      and m in cfg.SUBJECTS]
            steps = []
            for lo, hi in zip(ladder, ladder[1:]):
                a, b = metrics[lo][outcome], metrics[hi][outcome]
                draws = b["_draws"] - a["_draws"]
                steps.append({"from": lo, "to": hi, "diff": b["value"] - a["value"], "ci": ci(draws),
                              "underpowered": bool(a.get("underpowered") or b.get("underpowered"))})
            jump = None
            for i, s in enumerate(steps):
                excludes_zero = np.isfinite(s["ci"][0]) and (s["ci"][0] > 0 or s["ci"][1] < 0)
                largest = all(abs(s["diff"]) > abs(o["diff"]) for j, o in enumerate(steps) if j != i)
                if excludes_zero and largest and not s["underpowered"]:
                    jump = f"{s['from']} -> {s['to']}"
            out[f"{outcome}|{family}"] = {"steps": steps, "jump": jump}
        if "Qwen3.5-27B" in metrics and "Qwen3.8-27B" in metrics:
            a, b = metrics["Qwen3.5-27B"][outcome], metrics["Qwen3.8-27B"][outcome]
            out[f"{outcome}|version_27B"] = {"from": "Qwen3.5-27B", "to": "Qwen3.8-27B", "diff": b["value"] - a["value"],
                                              "ci": ci(b["_draws"] - a["_draws"]), "exploratory": True}
    return out


# --- Figures ----------------------------------------------------------------------------------------------


def bar_positions(models: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    labels = [m + ("\n(stage 1 only)" if m in cfg.ADDON_SUBJECTS else "") for m in models]
    colors = [FAMILY_COLORS[cfg.ALL_SUBJECTS[m]["family"]] for m in models]
    return np.arange(len(models)), labels, colors


def err(entry: dict) -> tuple[float, float]:
    lo, hi = entry["ci"]
    return entry["value"] - lo, hi - entry["value"]


def family_legend(ax, models, extra=()):
    from matplotlib.patches import Patch
    fams = sorted({cfg.ALL_SUBJECTS[m]["family"] for m in models}, key=list(FAMILY_COLORS).index)
    handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in fams] + list(extra)
    ax.legend(handles=handles, loc="upper left", fontsize=8)


def fig_vea(metrics, models, path):
    x, labels, colors = bar_positions(models)
    fig, ax = plt.subplots(figsize=(1.0 * len(models) + 3, 4.5))
    bottom = np.zeros(len(models))
    shades = {"safety": 1.0, "capabilities": 0.7, "both": 0.5, "generic": 0.3}
    for f in FRAMINGS:
        vals = np.array([metrics[m][f"M2_rate_{f}"]["value"] for m in models]) * 100
        ax.bar(x, vals, bottom=bottom, color=colors, alpha=shades[f], edgecolor="k", linewidth=0.3,
               label=f"framing: {f}" if f == FRAMINGS[0] else None)
        for xi, b, v in zip(x, bottom, vals):
            if v > 0.6:
                ax.text(xi, b + v / 2, f[:3], ha="center", va="center", fontsize=6)
        bottom += vals
    totals = [metrics[m]["M1_vea_rate"] for m in models]
    ax.errorbar(x, [t["value"] * 100 for t in totals], yerr=np.array([err(t) for t in totals]).T * 100,
                fmt="none", ecolor="k", capsize=3)
    bloom = [(xi, cfg.BLOOM_A1_QWEN3_VEA[m] * 100) for xi, m in zip(x, models) if m in cfg.BLOOM_A1_QWEN3_VEA]
    if bloom:
        ax.scatter(*zip(*bloom), marker="D", color="k", zorder=5, label="Bloom et al. A1 (GPT-5-mini judge)")
    ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("VEA rate, % of scored rollouts (judge)")
    ax.set_title("F1: verbalized eval awareness on Fortress (stage 1), stacked by framing\n"
                 "(saf/cap/bot/gen = safety/capabilities/both/generic); 95% CI over questions", fontsize=10)
    family_legend(ax, models, extra=ax.get_legend_handles_labels()[0][-1:] if bloom else [])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_bars(ax, models, series: list[tuple[str, list[dict | None]]], width=0.8):
    x, labels, colors = bar_positions(models)
    k = len(series)
    for j, (name, entries) in enumerate(series):
        offs = x - width / 2 + (j + 0.5) * width / k
        for xi, e, c in zip(offs, entries, colors):
            if e is None or not np.isfinite(e["value"]):
                continue
            alpha = 1 - 0.6 * j / max(k - 1, 1)
            underpowered = bool(e.get("underpowered"))
            ax.bar(xi, e["value"] * 100, width / k, color=c, alpha=alpha * (0.35 if underpowered else 1),
                   hatch="//" if underpowered else None, edgecolor="k", linewidth=0.3)
            if not underpowered:
                ax.errorbar(xi, e["value"] * 100, yerr=np.nan_to_num(np.array([err(e)]).T * 100), fmt="none", ecolor="k", capsize=2)
        ax.bar(np.nan, np.nan, color="gray", alpha=1 - 0.6 * j / max(k - 1, 1), label=name)
    ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=7)


def fig_refusal(metrics, models, path):
    fig, ax = plt.subplots(figsize=(1.0 * len(models) + 3, 4.5))
    grouped_bars(ax, models, [("adversarial (stage 1)", [metrics[m]["M3_refusal_adversarial"] for m in models]),
                              ("benign twin", [metrics[m]["M3_refusal_benign"] for m in models])])
    ax.set_ylabel("refusal rate, % (judge)")
    ax.set_title("F2: refusal on Fortress adversarial prompts vs their benign twins; 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_within(metrics, models, path):
    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 3, 4.5))
    grouped_bars(ax, models, [
        ("pooled VEA vs non-VEA", [metrics[m]["M4_within_question_pooled"] for m in models]),
        ("safety-framed VEA vs non-VEA", [metrics[m]["M4_framing_safety"] for m in models]),
        ("capabilities-framed VEA vs non-VEA", [metrics[m]["M4_framing_capabilities"] for m in models]),
        ("placebo (jailbreak-noticing) vs neither", [metrics[m]["M6_placebo_within_question"] for m in models]),
    ])
    ax.set_ylabel("within-question refusal difference, pp (judge)")
    ax.set_title("F3: within-question refusal difference (Mantel-Haenszel); 95% CI\n"
                 "faded hatched bars: < 20 informative questions (underpowered, no CI shown)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_across(metrics, models, path):
    x, labels, colors = bar_positions(models)
    fig, axes = plt.subplots(1, 2, figsize=(2 * len(models) + 3, 4.5))
    for ax, key, ylabel, scale in ((axes[0], "M5a_gap_nonvea_refusal", "refusal gap, pp", 100),
                                   (axes[1], "M5b_spearman_v_r0", "Spearman rho", 1)):
        entries = [metrics[m][key] for m in models]
        vals = [e["value"] * scale for e in entries]
        ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.3)
        ax.errorbar(x, vals, yerr=np.nan_to_num(np.array([err(e) for e in entries]).T * scale), fmt="none", ecolor="k", capsize=3)
        ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(ylabel + " (judge)")
    axes[0].set_title("F4a: non-VEA refusal, questions with >= 1 VEA rollout in 10 minus none")
    axes[1].set_title("F4b: Spearman(VEA share, non-VEA refusal) across questions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_auroc(metrics, models, path):
    have = [m for m in models if f"M7_auroc_vs_{cfg.LOGIT_MONITOR_CONTRASTS[0]}" in metrics[m]]
    if not have:
        return
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * len(have) + 3, 4.8))
    primary = f"M7_auroc_vs_{cfg.LOGIT_MONITOR_CONTRASTS[0]}"
    names = [("primary", "raw score"), ("contrast", "contrast score"),
             ("primary_length_residualized", "raw, length-residualized"), ("length_only_baseline", "prompt length only")]
    grouped_bars(axes[0], have, [(label, [ {**metrics[m][primary][k], "value": metrics[m][primary][k]["value"]} for m in have])
                                 for k, label in names])
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("AUROC x 100")
    axes[0].set_title(f"F5a: logit monitor, Fortress adversarial vs {cfg.LOGIT_MONITOR_CONTRASTS[0]}")
    grouped_bars(axes[1], have, [(f"vs {o}", [metrics[m].get(f"M7_auroc_vs_{o}", {}).get("primary") for m in have])
                                 for o in cfg.LOGIT_MONITOR_CONTRASTS])
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("F5b: raw score AUROC, Fortress adversarial vs each set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Samples and labeling sheets ------------------------------------------------------------------------


def read_samples(tables: dict[str, pd.DataFrame], texts: dict, path: Path, smoke: bool) -> None:
    rng = random.Random(cfg.READ_SAMPLE_SEED)
    lines = [f"# {cfg.EXP_ID}: read samples", "",
             f"Random stage-1 Fortress adversarial rollouts, `random.Random({cfg.READ_SAMPLE_SEED})`: 3 per model "
             "drawn from all scored rollouts, then 2 drawn from the judge's VEA-positive rollouts (labeled as such). "
             "CoT shown in full up to 4,000 characters; answer = the judged first 384 tokens.", ""]
    for model, df in tables.items():
        pool = df[(df.prompt_set == "fortress_adv") & (df.stage == "stage1") & df.scored]
        picks = [("random", r) for r in pool.sample(3, random_state=rng.randint(0, 2**31)).itertuples()] if len(pool) >= 3 else []
        vea = pool[pool.V == 1]
        picks += [("judge VEA-positive", r) for r in vea.sample(min(2, len(vea)), random_state=rng.randint(0, 2**31)).itertuples()]
        lines.append(f"## {model}")
        rollouts = {ladder_judge.rollout_key(r): r for r in ladder_judge.load_rollouts(model, smoke)}
        for kind, r in picks:
            ro = rollouts[r.rollout_key]
            lines += [f"### {kind}: question {r.item_id}, rollout {r.rollout_index} | judge V={r.V:.0f} framing={r.framing} "
                      f"P={r.P:.0f} R={r.R:.0f} | CoT {r.n_cot_tokens} tokens",
                      "**Prompt (first 600 chars):**", "", "```", texts[("fortress_adv", r.item_id)][:600], "```",
                      "**CoT:**", "", "```", ro["cot_text"][:4000], "```", "**Answer:**", "", "```", ro["answer_text"][:2500], "```", ""]
    path.write_text("\n".join(lines) + "\n")


def labeling_sheet(tables: dict[str, pd.DataFrame], texts: dict, sheet_id: int, out_dir: Path, smoke: bool) -> None:
    """Blinded sheet (no model name, no judge label): per model 8 judge-VEA, 8 regex-positive judge-non-VEA
    (the least confident first), 4 regex-negative; the key file maps item uids back."""
    rng = random.Random(f"{cfg.LABELING_SEED}|{sheet_id}")
    items = []
    for model, df in tables.items():
        pool = df[(df.prompt_set == "fortress_adv") & df.V.notna() & df.status.isin(["scored", "truncated"])]
        conf = (pool.p_V.fillna(0.5) - 0.5).abs()
        groups = [("judge_vea", pool[pool.V == 1], 8), ("regex_pos_judge_nonvea", pool[pool.regex_hit & (pool.V == 0)], 8),
                  ("regex_negative", pool[~pool.regex_hit], 4)]
        for stratum, g, n in groups:
            if not len(g):
                continue
            if stratum != "regex_negative":
                uncertain = g.loc[conf[g.index].sort_values().index[: n // 2]]
                rest = g.drop(uncertain.index)
                picked = pd.concat([uncertain, rest.sample(min(n - len(uncertain), len(rest)), random_state=rng.randint(0, 2**31))])
            else:
                picked = g.sample(min(n, len(g)), random_state=rng.randint(0, 2**31))
            items += [(model, stratum, r) for r in picked.itertuples()]
    rng.shuffle(items)
    rollouts = {}
    for model in tables:
        rollouts.update({ladder_judge.rollout_key(r): r for r in ladder_judge.load_rollouts(model, smoke)})
    sheet, key = [], []
    for i, (model, stratum, r) in enumerate(items):
        uid = f"s{sheet_id}-{i:03d}"
        ro = rollouts[r.rollout_key]
        sheet.append({"uid": uid, "prompt": texts[("fortress_adv", r.item_id)][:3000], "cot": ro["cot_text"][:30000],
                      "answer": ro["answer_text"], "human_vea (1/0)": "", "human_refusal (1/0)": "", "notes": ""})
        key.append({"uid": uid, "model": model, "stratum": stratum, "rollout_key": r.rollout_key,
                    "judge_V": r.V, "judge_p_V": r.p_V, "judge_framing": r.framing, "judge_R": r.R})
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path = out_dir / f"sheet_{sheet_id}.csv", out_dir / f"key_{sheet_id}.json"
    for p in (sheet_path, key_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; not overwriting (a human may have filled it in)")
    pd.DataFrame(sheet).to_csv(sheet_path, index=False)
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    print(f"labeling sheet {sheet_id}: {len(sheet)} items -> {sheet_path}")


def hand_label_agreement(out_dir: Path) -> dict:
    out = {}
    for sheet_path in sorted(out_dir.glob("sheet_*_filled.csv")):
        sheet_id = sheet_path.stem.split("_")[1]
        key = {k["uid"]: k for k in json.loads((out_dir / f"key_{sheet_id}.json").read_text())}
        filled = pd.read_csv(sheet_path)
        rows = [(key[u]["model"], key[u]["judge_V"], h) for u, h in zip(filled.uid, filled["human_vea (1/0)"]) if pd.notna(h)]
        if rows:
            j = np.array([r[1] for r in rows], dtype=float)
            h = np.array([r[2] for r in rows], dtype=float)
            out[f"sheet_{sheet_id}"] = {"n": len(rows), "kappa_vea": kappa(j, h),
                                        "judge_precision": float(h[j == 1].mean()) if (j == 1).any() else None,
                                        "judge_recall_on_sheet": float(j[h == 1].mean()) if (h == 1).any() else None,
                                        "note": "stratified sheet (over-samples positives and uncertain items)"}
    return out


# --- Main -----------------------------------------------------------------------------------------------------


def strip_draws(obj):
    if isinstance(obj, dict):
        return {k: strip_draws(v) for k, v in obj.items() if k != "_draws"}
    if isinstance(obj, list):
        return [strip_draws(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def flat_row(model: str, m: dict) -> dict:
    row = {"model": model, "family": cfg.ALL_SUBJECTS[model]["family"], "params_b": cfg.ALL_SUBJECTS[model]["params_b"]}
    for k, v in m.items():
        if isinstance(v, dict) and "value" in v and "ci" in v:
            row[k] = v["value"]
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = v["ci"]
            if "underpowered" in v:
                row[f"{k}_underpowered"] = v["underpowered"]
        elif isinstance(v, (int, float)):
            row[k] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", required=True, help="'final' writes to results/exp02_vea_ladder/")
    parser.add_argument("--models", nargs="+", choices=list(cfg.ALL_SUBJECTS))
    parser.add_argument("--labeling-sheet", type=int, help="also write blinded labeling sheet N")
    parser.add_argument("--smoke", action="store_true", help="analyse the smoke-test caches (code test only)")
    args = parser.parse_args()
    if args.smoke and args.out_name == "final":
        parser.error("--smoke cannot write the final results")

    smoke = args.smoke
    models = available_models(args.models, smoke)
    out_dir = cfg.RESULTS_DIR if args.out_name == "final" else cfg.RESULTS_DIR / args.out_name
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} exists; results are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = ladder_judge.prompt_texts()
    questions = sorted({i for (s, i) in texts if s == "fortress_adv"}, key=int)
    boot = QuestionBootstrap(questions)
    tables = {m: load_rollout_table(m, smoke) for m in models}
    metrics = {m: model_metrics(tables[m], load_logit(m, smoke), boot) for m in models}
    # Logit monitor for models without generation (e.g. Qwen3-32B if stage 1 was skipped).
    logit_only = {}
    for m in cfg.RUN_ORDER:
        if m not in models and (lg := load_logit(m, smoke)) is not None:
            logit_only[m] = logit_metrics(lg, pd.DataFrame(columns=["v"]))

    jumps = jump_tests(metrics, models)
    checks = judge_checks(models, smoke)
    checks["hand_labels"] = {} if smoke else hand_label_agreement(cfg.LABELING_DIR)
    judge_swap = {m: {"ours": metrics[m]["M1_vea_rate"]["value"], "ours_ci": metrics[m]["M1_vea_rate"]["ci"],
                      "bloom_a1": cfg.BLOOM_A1_QWEN3_VEA[m]} for m in models if m in cfg.BLOOM_A1_QWEN3_VEA}

    tag = "" if args.out_name == "final" else f"{args.out_name}_"
    figures = {
        "F1": cfg.figure_path(f"{tag}F1_vea_rate"),
        "F2": cfg.figure_path(f"{tag}F2_refusal"),
        "F3": cfg.figure_path(f"{tag}F3_within_question"),
        "F4": cfg.figure_path(f"{tag}F4_across_question"),
        "F5": cfg.figure_path(f"{tag}F5_logit_monitor_auroc"),
    }
    for p in figures.values():
        if p.exists():
            raise FileExistsError(f"{p} exists; figures are never overwritten")
    fig_vea(metrics, models, figures["F1"])
    fig_refusal(metrics, models, figures["F2"])
    fig_within(metrics, models, figures["F3"])
    fig_across(metrics, models, figures["F4"])
    fig_auroc({**metrics, **logit_only}, [m for m in cfg.RUN_ORDER if m in metrics or m in logit_only], figures["F5"])

    summary = strip_draws({
        "exp_id": cfg.EXP_ID, "out_name": args.out_name, "models": models,
        "evidence_note": "judge-labeled numbers (V, R, P and everything derived) carry the 'judge' tag; "
                         "M7 AUROCs are judge-free",
        "primary_outcomes": cfg.PRIMARY_OUTCOMES,
        "per_model": metrics, "logit_monitor_only": logit_only, "jump_tests": jumps,
        "judge_checks": checks, "judge_swap_vs_bloom_a1": judge_swap,
        "figures": {k: str(v.relative_to(cfg.REPO_ROOT)) for k, v in figures.items()},
        "bootstrap": {"unit": "Fortress question", "iters": cfg.BOOTSTRAP_ITERS, "seed": cfg.BOOTSTRAP_SEED, "shared_across_models": True},
    })
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.DataFrame([flat_row(m, metrics[m]) for m in models]).to_csv(out_dir / cfg.PER_MODEL_CSV_PATH.name, index=False)
    read_samples(tables, texts, out_dir / cfg.READ_SAMPLES_PATH.name, smoke)
    if args.labeling_sheet:
        labeling_sheet(tables, texts, args.labeling_sheet, out_dir / "labeling" if smoke else cfg.LABELING_DIR, smoke)
    print(f"wrote {summary_path}")
    for m in models:
        e = metrics[m]
        print(f"{m:14s} VEA {e['M1_vea_rate']['value']:.3f} {e['M1_vea_rate']['ci']}  refusal {e['M3_refusal_adversarial']['value']:.3f}  "
              f"M4 {e['M4_within_question_pooled']['value']:+.3f} {e['M4_within_question_pooled']['ci']} "
              f"(n_inf {e['M4_within_question_pooled']['n_informative_questions']})")


if __name__ == "__main__":
    main()
