"""Metrics per variant and paired differences against the same model's bf16 baseline.

    uv run python -m eval_awareness.quant.analyze            # every variant in QUANT_DATA/rollouts
    uv run python -m eval_awareness.quant.analyze --smoke    # the *.smoke files only

Per variant: the unified metrics (eval_awareness.unified.metrics) on rollouts 0-9, plus truncation, reasoning
length, VEA per 1,000 reasoning tokens, and benign-twin refusal. Paired: for each quantized variant, the change
from <base>@bf16 in refusal, VEA and the within-question refusal gap, over questions present in both, with 95%
percentile CIs from 2,000 bootstrap draws that resample the same questions in both conditions. Refusal bounds
treat unscored rollouts (truncated, empty answer) as all refusing or none refusing.
Writes results/quant[/smoke]/summary.json, per_variant.csv, paired.csv and REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_awareness import unified as U
from eval_awareness.quant import config as qc

OUT = qc.ROOT / "results" / "quant"


def variants(smoke: bool) -> list[str]:
    names = sorted(p.name.removesuffix(".adversarial.jsonl") for p in (qc.DATA / "rollouts").glob("*.adversarial.jsonl"))
    return [n for n in names if n.endswith(".smoke") == smoke]


def table(variant: str, split: str = "adversarial") -> pd.DataFrame | None:
    df = U.load_table(variant, split, root=qc.DATA)
    return None if df is None else U.design(df, "first10")


def extras(df: pd.DataFrame, benign: pd.DataFrame | None) -> dict:
    s = df[df.status == "scored"]
    n_vea = int((s.vea == True).sum())  # noqa: E712
    out = {"truncation_rate": float((df.status == "truncated").mean()),
           "unscored_rate": float((df.status != "scored").mean()),
           "mean_cot_tokens_all": float(df.n_cot_tokens.mean()),
           "mean_cot_tokens_scored": float(s.n_cot_tokens.mean()) if len(s) else np.nan,
           "vea_per_1k_cot_tokens": float(1000 * n_vea / s.n_cot_tokens.sum()) if len(s) else np.nan}
    r = df.refusal.astype(float)
    unscored = df.status != "scored"
    out["refusal_bounds_unscored"] = [float(r.fillna(0).where(~unscored, 0).mean()),
                                      float(r.fillna(0).where(~unscored, 1).mean())]
    if benign is not None:
        b = benign[(benign.status == "scored") & benign.refusal.notna()]
        out["benign_refusal_rate"] = float(b.refusal.astype(float).mean()) if len(b) else np.nan
        out["benign_n"] = int(len(b))
    return out


def question_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per question: scored rollouts with both labels -> n, VEA count, refusals among VEA / non-VEA, and the
    count of rollouts with each label for the pooled rates."""
    s = df[df.status == "scored"]
    both = s[s.vea.notna() & s.refusal.notna()].assign(v=lambda x: x.vea.astype(float), r=lambda x: x.refusal.astype(float))
    return U.per_question(both)


def paired(base_df: pd.DataFrame, var_df: pd.DataFrame) -> dict:
    a, b = question_stats(base_df), question_stats(var_df)
    qs = a.index.intersection(b.index)
    a, b = a.loc[qs], b.loc[qs]
    cols = ["n", "n1", "n0", "r1", "r0"]
    stats = np.hstack([a[cols].to_numpy(float), b[cols].to_numpy(float)])

    def refusal(x):
        return (x[3].sum() + x[4].sum()) / x[0].sum()

    def vea(x):
        return x[1].sum() / x[0].sum()

    fns = {"refusal": lambda m: refusal(m[5:]) - refusal(m[:5]),
           "vea": lambda m: vea(m[5:]) - vea(m[:5]),
           "within_q_gap": lambda m: U.post_diff_from(m[5:]) - U.post_diff_from(m[:5])}
    out = {"n_questions": int(len(qs))}
    for name, fn in fns.items():
        out[f"delta_{name}"] = float(fn(stats.T))
        out[f"delta_{name}_ci"] = U.boot_ci(stats, fn)
    return out


W4_SEEDS = ["gptq-w4", "gptq-w4-calB", "gptq-w4-calC"]  # same recipe, calibration rows 0:1k, 1k:2k, 2k:3k


def seed_summary(tables: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per base model with bf16 and all three W4 calibration seeds: each seed's change from bf16, the mean and SD over
    seeds, and the largest seed-to-seed difference; then the mean W4 change pooled over bases. CIs come from one
    bootstrap that resamples Fortress questions jointly (the same qids for every model), so they reflect question
    sampling given these checkpoints; the SD over the three seeds is the checkpoint-to-checkpoint spread."""
    cols = ["n", "n1", "n0", "r1", "r0"]
    bases = sorted({qc.split_variant(v)[0] for v in tables
                    if all(f"{qc.split_variant(v)[0]}@{c}" in tables for c in ["bf16", *W4_SEEDS])})
    stats, qsets = {}, []
    for b in bases:
        for c in ["bf16", *W4_SEEDS, "gptq-w8", "hf-w4"]:
            if f"{b}@{c}" in tables:
                stats[(b, c)] = question_stats(tables[f"{b}@{c}"])[cols]
                qsets.append(set(stats[(b, c)].index))
    qs = sorted(set.intersection(*qsets)) if qsets else []
    arr = {k: v.loc[qs].to_numpy(float) for k, v in stats.items()}

    def rates(m, idx):  # (refusal, vea) pooled over the questions idx
        x = m[idx]
        return (x[:, 3].sum() + x[:, 4].sum()) / x[:, 0].sum(), x[:, 1].sum() / x[:, 0].sum()

    def deltas(idx):
        d = {}
        for b in bases:
            r0, v0 = rates(arr[(b, "bf16")], idx)
            for c in [*W4_SEEDS, "gptq-w8", "hf-w4"]:
                if (b, c) in arr:
                    r, v = rates(arr[(b, c)], idx)
                    d[(b, c)] = (r - r0, v - v0)
        return d

    full = np.arange(len(qs))
    point = deltas(full)
    rng = np.random.default_rng(seed)
    boots = [deltas(rng.integers(0, len(qs), len(qs))) for _ in range(n_boot)]

    def ci(f):
        xs = np.array([f(d) for d in boots])
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

    out = {"n_questions": len(qs), "bases": {}, "pooled": {}}
    for b in bases:
        e = {}
        for k, name in ((0, "refusal"), (1, "vea")):
            seeds = [point[(b, c)][k] for c in W4_SEEDS]
            e[name] = {"w4_seeds": seeds, "w4_mean": float(np.mean(seeds)), "w4_sd": float(np.std(seeds, ddof=1)),
                       "w4_max_seed_gap": float(max(seeds) - min(seeds)),
                       "w4_mean_ci": ci(lambda d, b=b, k=k: np.mean([d[(b, c)][k] for c in W4_SEEDS]))}
            for c in ("gptq-w8", "hf-w4"):
                if (b, c) in point:
                    e[name][c] = point[(b, c)][k]
                    e[name][c + "_ci"] = ci(lambda d, b=b, c=c, k=k: d[(b, c)][k])
        out["bases"][b] = e
    for k, name in ((0, "refusal"), (1, "vea")):
        f = lambda d, k=k: np.mean([np.mean([d[(b, c)][k] for c in W4_SEEDS]) for b in bases])  # noqa: E731
        w8 = [b for b in bases if (b, "gptq-w8") in point]
        g = lambda d, k=k: np.mean([d[(b, "gptq-w8")][k] for b in w8])  # noqa: E731
        out["pooled"][name] = {"w4_mean_over_bases": float(f(point)), "w4_mean_over_bases_ci": ci(f),
                               "n_bases_negative_w4_mean": int(sum(out["bases"][b][name]["w4_mean"] < 0 for b in bases)),
                               "n_bases": len(bases),
                               "w8_mean_over_bases": float(g(point)) if w8 else None,
                               "w8_mean_over_bases_ci": ci(g) if w8 else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = OUT / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, summary, rows = {}, {}, []
    for v in variants(args.smoke):
        df = table(v)
        benign = None if args.smoke else table(v, "benign")
        tables[v] = df
        m = U.metrics(df) | extras(df, benign)
        summary[v] = m
        rows.append({"variant": v, **{k: m.get(k) for k in (
            "n_rollouts", "n_assigned", "label_coverage", "vea_rate", "refusal_rate", "within_q_refusal_diff",
            "n_informative_questions", "truncation_rate", "mean_cot_tokens_scored", "vea_per_1k_cot_tokens",
            "benign_refusal_rate", "vea_rate_ci", "refusal_rate_ci", "within_q_refusal_diff_ci")}})

    prows = []
    for v, df in tables.items():
        name = v.removesuffix(".smoke")
        base, condition = qc.split_variant(name)
        ref = f"{base}@bf16" + (".smoke" if args.smoke else "")
        if condition == "bf16" or ref not in tables:
            continue
        p = paired(tables[ref], df)
        summary[v]["paired_vs_bf16"] = p
        prows.append({"variant": v, "base": base, "condition": condition, **p})

    seeds = None if args.smoke else seed_summary(tables)
    if seeds:
        (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.DataFrame(rows).to_csv(out_dir / "per_variant.csv", index=False)
    pd.DataFrame(prows).to_csv(out_dir / "paired.csv", index=False)
    report(out_dir, summary, args.smoke, seeds)
    print(f"wrote {out_dir}")


def pct(x, ci=None) -> str:
    if x is None or x != x:
        return "-"
    s = f"{100 * x:.1f}"
    return s + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci and ci[0] == ci[0] else "")


def pp(x, ci) -> str:
    return "-" if x != x else f"{100 * x:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def report(out_dir: Path, summary: dict, smoke: bool, seeds: dict | None = None) -> None:
    md = [f"# Quantization study{' (smoke)' if smoke else ''}", "",
          "Judges: unified v2 (vea_cal and F2 refusal, Qwen3.8-27B bf16). Rollouts 0-9 per Fortress question "
          "(smoke: 20 questions x 4). Rates over scored rollouts with a valid label; 95% CIs resample questions.", "",
          "| variant | scored / assigned | label coverage | VEA % | refusal % | within-q gap pp | truncated % | "
          "mean CoT tokens | VEA per 1k CoT tokens | benign refusal % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, m in summary.items():
        md.append(f"| {v} | {m['n_rollouts']} / {m['n_assigned']} | {pct(m['label_coverage'])} | "
                  f"{pct(m['vea_rate'], m.get('vea_rate_ci'))} | {pct(m['refusal_rate'], m.get('refusal_rate_ci'))} | "
                  f"{pct(m['within_q_refusal_diff'], m.get('within_q_refusal_diff_ci'))} | {pct(m['truncation_rate'])} | "
                  f"{m['mean_cot_tokens_scored']:.0f} | {m['vea_per_1k_cot_tokens']:.2f} | {pct(m.get('benign_refusal_rate'))} |")
    md += ["", "## Paired change vs the same model's bf16 baseline (percentage points)", "",
           "| variant | questions | Δ refusal | Δ VEA | Δ within-q gap |", "|---|---|---|---|---|"]
    for v, m in summary.items():
        p = m.get("paired_vs_bf16")
        if p:
            md.append(f"| {v} | {p['n_questions']} | {pp(p['delta_refusal'], p['delta_refusal_ci'])} | "
                      f"{pp(p['delta_vea'], p['delta_vea_ci'])} | {pp(p['delta_within_q_gap'], p['delta_within_q_gap_ci'])} |")
    if seeds and seeds["bases"]:
        md += ["", f"## W4 calibration seeds vs bf16 ({seeds['n_questions']} questions common to all variants, pp)", "",
               "Three GPTQ W4 checkpoints per model, same recipe, calibration rows 0:1k / 1k:2k / 2k:3k. CIs resample "
               "questions jointly; SD is over the three seeds.", ""]
        for name in ("refusal", "vea"):
            md += [f"### {name}", "", "| model | seed A | seed B | seed C | mean [95% CI] | SD | W8 | off-the-shelf W4 |",
                   "|---|---|---|---|---|---|---|---|"]
            for b, e in seeds["bases"].items():
                x = e[name]
                s = " | ".join(f"{100 * v:+.1f}" for v in x["w4_seeds"])
                w8 = pp(x["gptq-w8"], x["gptq-w8_ci"]) if "gptq-w8" in x else "-"
                hf = pp(x["hf-w4"], x["hf-w4_ci"]) if "hf-w4" in x else "-"
                md.append(f"| {b} | {s} | {pp(x['w4_mean'], x['w4_mean_ci'])} | {100 * x['w4_sd']:.1f} | {w8} | {hf} |")
            p = seeds["pooled"][name]
            md += ["", f"Pooled over {p['n_bases']} models: W4 seed-mean {pp(p['w4_mean_over_bases'], p['w4_mean_over_bases_ci'])}"
                   f" ({p['n_bases_negative_w4_mean']} of {p['n_bases']} models negative); W8 "
                   + (pp(p['w8_mean_over_bases'], p['w8_mean_over_bases_ci']) if p["w8_mean_over_bases"] is not None else "-"), ""]
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
