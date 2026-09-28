"""Gemma 4 follow-up: judge calibration scores and the VEA-vs-refusal analysis on prompts with VEA.

    uv run python -m eval_awareness.followup calib
    uv run python -m eval_awareness.followup analyze --task vea_cal
    uv run python -m eval_awareness.followup smoke --slug NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
        --dir results/nemotron3_super --rollouts results/nemotron3_super/rollouts

Defaults are the Gemma 4 follow-up's paths; --slug/--dir/--rollouts/--refusal-dir point it at another model.

calib: scores each judge variant in results/gemma4_followup/calibration/<variant>/ against the
hand labels in calibration_set.jsonl (labels by Claude from reading the CoTs; see REPORT.md).

analyze: on the adversarial prompts in vea_qids.txt (every prompt with >=1 VEA rollout in the pilot,
q285 excluded), compares refusal in VEA vs non-VEA rollouts of the same prompt, and correlates
per-prompt VEA rate with refusal rate. Also reports the VEA rate on GSM8K and MATH-500.
CIs are 95% percentile intervals from a bootstrap over prompts (seed BOOT_SEED, N_BOOT draws).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eval_awareness.io import read_jsonl

BOOT_SEED, N_BOOT = 20260926, 2000
# overridden from the command line in main()
F = Path("results/gemma4_followup")
R = Path("results/gemma4_pilot/rollouts")
REFUSAL_DIR = Path("results/gemma4_pilot/judgments")
SLUG = "gemma-4-31B-it"
JUDGE = "Qwen3.8-27B"



def latest(rows: list[dict], key=("qid", "sample_idx")) -> pd.DataFrame:
    if not rows:  # missing or still-empty file
        return pd.DataFrame(columns=[*key, "vea", "quote", "refusal"])
    return pd.DataFrame(rows).drop_duplicates(list(key), keep="last").reset_index(drop=True)


def calib() -> dict:
    labels = latest(read_jsonl(F / "calibration/calibration_set.jsonl"))[["qid", "label", "kind"]]
    out = {}
    for d in sorted(p for p in (F / "calibration").iterdir() if p.is_dir()):
        files = list(d.glob("calibration_set.*.jsonl"))
        files = [f for f in files if not f.name.endswith(".config.jsonl")]
        rows = read_jsonl(files[0]) if files else []
        if not rows:
            continue
        j = latest(rows).merge(labels, on="qid")
        pred, y = j.vea.fillna(False).astype(bool), j.label.astype(bool)
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        out[d.name] = {"n": len(j), "tp": tp, "fp": fp, "fn": fn, "tn": int((~pred & ~y).sum()),
                       "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                       "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                       "positive_rate_by_kind": j.groupby("kind").apply(lambda g: float(g.vea.fillna(False).mean()),
                                                                        include_groups=False).round(3).to_dict()}
    return out


def boot(values: np.ndarray, stat) -> list[float]:
    rng = np.random.default_rng(BOOT_SEED)
    s = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(N_BOOT)]
    s = np.asarray([x for x in s if x is not None and not np.isnan(x)])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def rate_ci(df: pd.DataFrame, col: str) -> dict:
    """Rollout-level rate with a prompt-clustered bootstrap CI."""
    g = df.groupby("qid")[col].agg(["sum", "count"]).to_numpy(float)
    return {"rate": float(g[:, 0].sum() / g[:, 1].sum()), "ci95": boot(g, lambda x: x[:, 0].sum() / x[:, 1].sum()),
            "n_rollouts": int(g[:, 1].sum()), "n_prompts": len(g), "n_positive": int(g[:, 0].sum())}


def within_stats(d: pd.DataFrame, col: str):
    """Per-prompt table, the within-prompt weighted refusal difference (all prompts with both kinds of rollout,
    and those whose refusal is not stuck at 0/1), and Spearman(VEA rate, refusal rate) across prompts."""
    per = d.groupby("qid").apply(lambda g: pd.Series({
        "N": len(g), "n_vea": int(g[col].sum()), "vea_rate": g[col].mean(), "refusal": g.refusal.mean(),
        "refusal_vea": g[g[col]].refusal.mean() if g[col].any() else np.nan,
        "refusal_no_vea": g[~g[col]].refusal.mean() if (~g[col]).any() else np.nan}), include_groups=False)
    both = per[(per.n_vea > 0) & (per.n_vea < per.N)].copy()
    both["diff"] = both.refusal_vea - both.refusal_no_vea
    both["w"] = both.n_vea * (both.N - both.n_vea)
    wdiff = lambda x: (x[:, 0] * x[:, 1]).sum() / x[:, 1].sum()
    arr = both[["diff", "w"]].to_numpy(float)
    varies = both[(both.refusal > 0) & (both.refusal < 1)][["diff", "w"]].to_numpy(float)
    pv = per[["vea_rate", "refusal"]].to_numpy(float)
    rho = lambda x: spearmanr(x[:, 0], x[:, 1]).statistic
    within = {"n_prompts": len(both), "weighted_diff": wdiff(arr) if len(arr) else None,
              "ci95": boot(arr, wdiff) if len(arr) else None,
              "n_prompts_positive_diff": int((both["diff"] > 0).sum()),
              "n_prompts_negative_diff": int((both["diff"] < 0).sum())}
    varies_within = {"n_prompts": len(varies), "weighted_diff": wdiff(varies) if len(varies) else None,
                     "ci95": boot(varies, wdiff) if len(varies) else None}
    rho_res = {"rho": float(rho(pv)), "ci95": boot(pv, rho), "p": float(spearmanr(pv[:, 0], pv[:, 1]).pvalue)}
    return per, both, within, varies_within, rho_res


def vea_types(split: str, task: str) -> dict | None:
    """Shares of VEA instances by the category judge's CATEGORY and TESTING subcategory (the post's taxonomy)."""
    c = latest(read_jsonl(F / "judgments" / f"{SLUG}.{split}.category.{JUDGE}.jsonl"))
    if c.empty:
        return None
    v = latest(read_jsonl(F / "judgments" / f"{SLUG}.{split}.{task}.{JUDGE}.jsonl"))
    c = c.merge(v[v.vea.fillna(False)][["qid", "sample_idx"]], on=["qid", "sample_idx"])  # current VEA positives
    sub = c.category.fillna("UNPARSED").where(c.category != "TESTING", "TESTING/" + c.subcategory.fillna("NONE"))
    return {"n": len(c), "counts": sub.value_counts().to_dict(),
            "shares": sub.value_counts(normalize=True).round(3).to_dict()}


def analyze(task: str) -> dict:
    jd = F / "judgments"
    res = {"judge_task": task, "boot": {"seed": BOOT_SEED, "n": N_BOOT, "cluster": "prompt"}}

    # Fortress: prompts with >=1 VEA in the pilot
    qf = F / "vea_qids.txt"
    if not qf.exists():  # default: every adversarial prompt with >=1 VEA rollout under this judge task
        vj = latest(read_jsonl(F / "judgments" / f"{SLUG}.adversarial.{task}.{JUDGE}.jsonl"))
        qf.write_text(" ".join(map(str, sorted(set(vj[vj.vea.fillna(False).astype(bool)].qid.astype(int))))) + "\n")
    qids = [int(x) for x in qf.read_text().split()]
    ro = latest(read_jsonl(R / f"{SLUG}.adversarial.jsonl"))
    ro = ro[ro.qid.isin(qids) & ~ro.thought_empty]
    v = latest(read_jsonl(jd / f"{SLUG}.adversarial.{task}.{JUDGE}.jsonl"))[["qid", "sample_idx", "vea", "quote"]]
    rf = latest(read_jsonl(REFUSAL_DIR / f"{SLUG}.adversarial.refusal.{JUDGE}.jsonl"))
    d = ro.merge(v, on=["qid", "sample_idx"]).merge(rf[["qid", "sample_idx", "refusal"]], on=["qid", "sample_idx"])
    d = d[d.refusal.notna() & d.vea.notna()].assign(vea=lambda x: x.vea.astype(bool), refusal=lambda x: x.refusal.astype(bool))

    per, both, within, varies_within, rho_res = within_stats(d, "vea")
    res["fortress"] = {
        "n_prompts": len(per), "n_rollouts": len(d), "rollouts_per_prompt": per.N.value_counts().to_dict(),
        "vea": rate_ci(d.assign(x=d.vea.astype(float)), "x"),
        "refusal": rate_ci(d.assign(x=d.refusal.astype(float)), "x"),
        "refusal_in_vea_rollouts": rate_ci(d[d.vea].assign(x=d[d.vea].refusal.astype(float)), "x"),
        "refusal_in_non_vea_rollouts": rate_ci(d[~d.vea].assign(x=d[~d.vea].refusal.astype(float)), "x"),
        "within_prompt": within,
        "within_prompt_refusal_varies": varies_within,
        "across_prompts_spearman_vea_rate_vs_refusal": rho_res,
        "per_prompt": per.reset_index().round(3).to_dict("records"),
    }

    res["vea_types_adversarial"] = vea_types("adversarial", task)

    # every adversarial prompt at the pilot's 20 rollouts, re-judged with this task (overall rate)
    allv = latest(read_jsonl(jd / f"{SLUG}.adversarial.{task}.{JUDGE}.jsonl"))
    allr = latest(read_jsonl(R / f"{SLUG}.adversarial.jsonl"))
    a20 = allr[(allr.sample_idx < 20) & ~allr.thought_empty].merge(allv[["qid", "sample_idx", "vea"]], on=["qid", "sample_idx"])
    a20 = a20[a20.vea.notna()]
    res["fortress_all_prompts_20_rollouts"] = {**rate_ci(a20.assign(x=a20.vea.astype(float)), "x"),
                                              "n_prompts_with_vea": int(a20.groupby("qid").vea.any().sum())}

    # capability benchmarks
    for split in ["gsm8k", "math500"]:
        ro = latest(read_jsonl(R / f"{SLUG}.{split}.jsonl"))
        vj = latest(read_jsonl(jd / f"{SLUG}.{split}.{task}.{JUDGE}.jsonl"))
        if ro.empty or vj.empty:
            continue
        c = ro[~ro.thought_empty].merge(vj[["qid", "sample_idx", "vea", "quote"]], on=["qid", "sample_idx"])
        c = c[c.vea.notna()]
        res[split] = {**rate_ci(c.assign(x=c.vea.astype(float)), "x"),
                      "n_prompts_with_vea": int(c.groupby("qid").vea.any().sum()),
                      "frac_truncated": float((ro.finish_reason == "length").mean()),
                      "mean_cot_tokens": float(ro.cot_tokens.mean()),
                      "quotes": c[c.vea][["qid", "sample_idx", "quote"]].to_dict("records"),
                      "vea_types": vea_types(split, task)}
    plot(both, per, res)
    return res


def plot(both: pd.DataFrame, per: pd.DataFrame, res: dict):
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid, surface = "#2a78d6", "#d9730d", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "font.size": 10, "figure.facecolor": surface, "axes.facecolor": surface})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    w = res["fortress"]["within_prompt"]
    title = f"Within prompt: {w['weighted_diff'] * 100:+.1f} pp [{w['ci95'][0] * 100:+.1f}, {w['ci95'][1] * 100:+.1f}]"
    if len(both) > 40:  # too many prompts to label: refusal with vs without VEA, one dot per prompt
        ax.plot([0, 1], [0, 1], color=grid, lw=1, zorder=1)
        ax.scatter(both.refusal_no_vea, both.refusal_vea, s=12 + 60 * both.w / both.w.max(), color=blue,
                   edgecolor=surface, lw=0.8, alpha=0.8, zorder=2)
        ax.set(xlabel="Refusal rate, non-VEA rollouts", ylabel="Refusal rate, VEA rollouts",
               xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), title=title + f"  (n={len(both)}, size = weight)")
        ax.spines[["top", "right"]].set_visible(False)
        return _right_panel(axes[1], per, res, fig, blue, grid, surface)
    b = both.sort_values("refusal_no_vea")
    y = np.arange(len(b))
    ax.hlines(y, b.refusal_no_vea, b.refusal_vea, color=grid, lw=2, zorder=1)
    ax.scatter(b.refusal_no_vea, y, s=28, color=muted, label="non-VEA rollouts", zorder=2)
    ax.scatter(b.refusal_vea, y, s=28, color=blue, label="VEA rollouts", zorder=3)
    ax.set_yticks(y, [f"q{q}" for q in b.index], fontsize=7)
    ax.set(xlabel="Refusal rate", xlim=(-0.03, 1.03), title=title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _right_panel(axes[1], per, res, fig, blue, grid, surface)


def _right_panel(ax, per, res, fig, blue, grid, surface):
    import matplotlib.pyplot as plt

    ax.scatter(per.vea_rate, per.refusal, s=36, color=blue, edgecolor=surface, lw=1.5)
    s = res["fortress"]["across_prompts_spearman_vea_rate_vs_refusal"]
    ax.set(xlabel="Per-prompt VEA rate", ylabel="Per-prompt refusal rate", ylim=(-0.03, 1.03),
           title=f"Across prompts: Spearman ρ = {s['rho']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "vea_refusal.png", dpi=150)
    plt.close(fig)


def smoke() -> dict:
    """Stage 0 checks for a <think>-style model: empty/unclosed thought blocks, truncation, seeds, throughput."""
    df = latest(read_jsonl(R / f"{SLUG}.adversarial.jsonl"))
    runs = read_jsonl(R / f"{SLUG}.adversarial.runs.jsonl")
    return {
        "n_rollouts": len(df), "n_prompts": int(df.qid.nunique()),
        "n_thought_empty": int(df.thought_empty.sum()), "n_thought_unclosed": int(df.response.isna().sum()),
        "n_response_empty": int((df.response.fillna("").str.strip() == "").sum()),
        "finish_reason": df.finish_reason.value_counts().to_dict(),
        "frac_truncated": float((df.finish_reason == "length").mean()),
        "cot_tokens": df.cot_tokens.describe().round(1).to_dict(),
        "response_tokens": df.response_tokens.describe().round(1).to_dict(),
        "seeds_unique": bool(df.seed.is_unique),
        "runs": [{k: r[k] for k in ["started", "gen_s", "n_rollouts", "output_tokens", "output_tok_per_s"]} for r in runs],
    }


def main():
    global F, R, SLUG, REFUSAL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "analyze", "smoke"])
    ap.add_argument("--task", default="vea_cal", help="VEA judge task whose judgments to analyze")
    ap.add_argument("--slug", default=SLUG, help="rollout file prefix (model slug)")
    ap.add_argument("--dir", type=Path, default=F, help="output dir; holds vea_qids.txt and judgments/")
    ap.add_argument("--rollouts", type=Path, default=R)
    ap.add_argument("--refusal-dir", type=Path, help="dir of refusal judgments (default: <dir>/judgments, or the "
                                                     "pilot's dir for the Gemma defaults)")
    args = ap.parse_args()
    F, R, SLUG = args.dir, args.rollouts, args.slug
    REFUSAL_DIR = args.refusal_dir or (REFUSAL_DIR if F == Path("results/gemma4_followup") else F / "judgments")
    if args.cmd == "smoke":
        out = smoke()
        (F / "smoke.json").write_text(json.dumps(out, indent=2))
    elif args.cmd == "calib":
        out = calib()
        (F / "calibration.json").write_text(json.dumps(out, indent=2))
    else:
        out = analyze(args.task)
        (F / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
        out = {k: v for k, v in out.items() if k not in ("gsm8k", "math500")} | \
              {k: {kk: vv for kk, vv in out[k].items() if kk != "quotes"} for k in ("gsm8k", "math500") if k in out}
        out["fortress"] = {k: v for k, v in out["fortress"].items() if k != "per_prompt"}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
