"""Compute the post's statistics from rollouts + judgments.

    uv run python -m eval_awareness.analyze --models Qwen3-0.6B [Qwen3-1.7B ...] --judge Qwen3-8B-4bit

Per model:
  * VEA rate per rollout (Fig A1.1), fraction of questions with >=1 VEA rollout, pass@k (Fig 3)
  * VEA category / testing subcategory shares and safety-relevant share (Fig A1.2)
  * refusal: within-question weighted difference with weights n(N-n) (App. C1),
    linear probability model with question random intercepts (App. C2),
    naive pooled difference (App. C3), refusal of questions with vs without any VEA (Fig 5),
    Spearman(#VEA rollouts, refusal) over questions with >=1 VEA (App. C6)
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eval_awareness.io import read_jsonl

SAFETY_SUBCATS = {"HARMFUL_COMPLIANCE", "BOUNDARIES"}


def load(model: str, judge: str, rollouts_dir="data/rollouts", judg_dir="data/judgments") -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(Path(rollouts_dir) / f"{model}.jsonl"))
    key = ["qid", "sample_idx"]
    for task, cols in [("vea", ["vea", "quote"]), ("refusal", ["refusal"]), ("category", ["category", "subcategory"])]:
        j = pd.DataFrame(read_jsonl(Path(judg_dir) / f"{model}.{task}.{judge}.jsonl"))
        if j.empty:
            continue
        j = j.drop_duplicates(key, keep="last")[key + [c for c in cols if c in j]]
        if task == "category":
            j = j.drop(columns=["quote"], errors="ignore")
        df = df.merge(j, on=key, how="left")
    return df


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: P(at least one of k draws without replacement is positive)."""
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def within_question_diff(df: pd.DataFrame) -> float | None:
    diffs, ws = [], []
    for _, g in df.groupby("qid"):
        n, N = int(g.vea.sum()), len(g)
        if 0 < n < N:
            diffs.append(g[g.vea].refusal.mean() - g[~g.vea].refusal.mean())
            ws.append(n * (N - n))
    return float(np.average(diffs, weights=ws)) if ws else None


def lpm_effect(df: pd.DataFrame) -> tuple[float, float] | None:
    import statsmodels.formula.api as smf

    d = df.assign(refusal=df.refusal.astype(float), vea=df.vea.astype(float))
    if d.vea.sum() == 0:
        return None
    fit = smf.mixedlm("refusal ~ vea", d, groups=d["qid"]).fit()
    return float(fit.params["vea"]), float(fit.bse["vea"])


def summarize(model: str, judge: str) -> dict:
    df = load(model, judge)
    out = {"model": model, "n_rollouts": len(df), "n_questions": df.qid.nunique(),
           "frac_truncated": float((df.finish_reason == "length").mean())}
    if "vea" not in df:
        return out
    df = df[df.vea.notna()].assign(vea=lambda d: d.vea.astype(bool))
    per_q = df.groupby("qid").vea.agg(["sum", "count"])
    out |= {
        "vea_rate": float(df.vea.mean()),
        "vea_rate_ci95": float(1.96 * np.sqrt(df.vea.mean() * (1 - df.vea.mean()) / len(df))),
        "frac_questions_any_vea": float((per_q["sum"] > 0).mean()),
        "pass_at_k": {k: float(np.mean([pass_at_k(int(r["count"]), int(r["sum"]), k) for _, r in per_q.iterrows()]))
                      for k in [1, 2, 5, 10, 20, 50, 100] if k <= per_q["count"].min()},
    }
    if "category" in df and df.vea.any():
        v = df[df.vea]
        out["category_counts"] = v.category.value_counts().to_dict()
        out["testing_subcategory_counts"] = v[v.category == "TESTING"].subcategory.value_counts().to_dict()
        out["frac_vea_safety_relevant"] = float(v.subcategory.isin(SAFETY_SUBCATS).mean())
    if "refusal" in df:
        r = df[df.refusal.notna()].assign(refusal=lambda d: d.refusal.astype(bool))
        anyq = r.groupby("qid").vea.transform("any")
        out |= {
            "n_refusal_graded": len(r),
            "refusal_rate": float(r.refusal.mean()),
            "refusal_vea": float(r[r.vea].refusal.mean()) if r.vea.any() else None,
            "refusal_no_vea": float(r[~r.vea].refusal.mean()),
            "within_question_diff": within_question_diff(r),
            "lpm_effect_se": lpm_effect(r),
            "refusal_questions_with_any_vea": float(r[anyq].groupby("qid").refusal.mean().mean()) if anyq.any() else None,
            "refusal_questions_without_vea": float(r[~anyq].groupby("qid").refusal.mean().mean()),
            "refusal_questions_with_any_vea_nonvea_rollouts_only":
                float(r[anyq & ~r.vea].groupby("qid").refusal.mean().mean()) if anyq.any() else None,
        }
        q = r.groupby("qid").agg(n_vea=("vea", "sum"), refusal=("refusal", "mean"))
        q = q[q.n_vea > 0]
        if len(q) > 2:
            rho, p = spearmanr(q.n_vea, q.refusal)
            out["spearman_nvea_refusal"] = {"rho": float(rho), "p": float(p), "N": len(q)}
    return out


def plot(summaries: list[dict], path: Path):
    import matplotlib.pyplot as plt

    s = [x for x in summaries if "vea_rate" in x]
    names = [x["model"].replace("Qwen3-", "") for x in s]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [100 * x["vea_rate"] for x in s],
                yerr=[100 * x["vea_rate_ci95"] for x in s], color="#4C72B0", capsize=4)
    axes[0].set(title="Verbalized eval awareness per rollout (Fortress)", ylabel="% of rollouts", xlabel="Qwen3 size")
    safe = [100 * x.get("frac_vea_safety_relevant", np.nan) for x in s]
    axes[1].bar(names, safe, color="#DD8452")
    axes[1].set(title="Safety-relevant share of VEA instances", ylabel="% of VEA instances", xlabel="Qwen3 size", ylim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B"])
    ap.add_argument("--judge", default="Qwen3-8B-4bit")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)
    summaries = [summarize(m, args.judge) for m in args.models]
    (args.out / f"summary.{args.judge}.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    plot(summaries, args.out / f"scaling.{args.judge}.png")


if __name__ == "__main__":
    main()
