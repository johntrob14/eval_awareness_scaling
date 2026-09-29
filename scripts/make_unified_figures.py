"""Figures for the unified results (results/unified/v2/figures/), linear scales only.

    uv run python scripts/make_unified_figures.py

Colors follow the model family in every figure (the dataviz reference palette, slots 1-5 in order):
Qwen3 blue, Qwen3.5 orange, Qwen3.8 aqua, Nemotron yellow, Gemma magenta. Every bar is labelled with its
model on the axis, so color never carries identity alone. Error bars are 95% CIs from 2,000 bootstrap draws
over questions unless a caption says otherwise.

Figures, with the post figure each mirrors where there is one:
  01_vea_rate                      VEA rate by model (post Fig. A1.1, all families)
  02_safety_share                  safety-relevant share of VEA (post Fig. A1.2)
  03_vea_types                     what the model says is being tested, stacked shares
  04_within_question_gap           refusal, VEA minus non-VEA rollouts of the same question (post Fig. 5 panel)
  05_refusal_by_question_vea       refusal on questions with / without any VEA rollout (post Fig. 5 split)
  06_nvea_vs_refusal               #VEA rollouts per question vs its refusal rate, Spearman rho (post App. C6)
  07_injection                     inserted VEA sentence minus none, next to the manipulation sentence
  08_removal                       keep minus remove, rejection-sampled, next to unfiltered
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_awareness import unified as U  # noqa: E402

OUT = U.RESULTS / "figures"
ORDER = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
         "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-122B-A10B",
         "Qwen3.5-397B-A17B", "Qwen3.8-27B", "Nemotron-3-Super", "gemma-4-31B-it"]
LABEL = {"Qwen3.5-122B-A10B": "Qwen3.5-122B", "Qwen3.5-397B-A17B": "Qwen3.5-397B",
         "Nemotron-3-Super": "Nemotron 3 Super", "gemma-4-31B-it": "Gemma 4 31B"}
FAMILY_COLOR = {"Qwen3": "#2a78d6", "Qwen3.5": "#eb6834", "Qwen3.8": "#1baf7a", "Nemotron": "#eda100", "Gemma": "#e87ba4"}
GREY, INK, INK2, GRID, SURFACE = "#bdbcb6", "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"


def family(m: str) -> str:
    return ("Qwen3.5" if m.startswith("Qwen3.5") else "Qwen3.8" if m.startswith("Qwen3.8") else
            "Qwen3" if m.startswith("Qwen3-") else "Nemotron" if m.startswith("Nemotron") else "Gemma")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    })


def positions(models):
    """x positions with a gap between families."""
    xs, x, prev = [], 0.0, None
    for m in models:
        if prev is not None and family(m) != family(prev):
            x += 0.7
        xs.append(x)
        x += 1
        prev = m
    return np.array(xs)


def finish(fig, ax, name, title, subtitle, ylabel, models, xs, legend=True, extra_handles=None):
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL.get(m, m) for m in models], rotation=40, ha="right", fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=10.5, color=INK2, va="top")
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()] if legend else []
    handles += extra_handles or []
    if handles:
        ax.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower left",
                  bbox_to_anchor=(0, 1.0), fontsize=9.5, handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.08, right=0.99)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {(OUT / f'{name}.png').relative_to(ROOT)}")


def bars_with_ci(ax, xs, vals, lo, hi, colors, width=0.72, labels=True, fmt="{:+.1f}", alpha=1.0, ci=True):
    ax.bar(xs, vals, width=width, color=colors, alpha=alpha, linewidth=0, zorder=2)
    if ci:
        ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                    ecolor=INK2, elinewidth=1.1, capsize=3, zorder=3)
    if labels:
        for x, v, h in zip(xs, vals, hi):
            ax.text(x, max(v, h) + 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0] or 1), fmt.format(v),
                    ha="center", va="bottom", fontsize=8.5, color=INK)


def main():
    style()
    S = json.loads((U.RESULTS / "summary.json").read_text())
    IV = json.loads((U.RESULTS / "interventions.json").read_text())
    models = [m for m in ORDER if m in S]
    xs = positions(models)
    colors = [FAMILY_COLOR[family(m)] for m in models]
    p = lambda m, k: S[m]["first10"][k]  # noqa: E731

    # 01 VEA rate
    fig, ax = plt.subplots(figsize=(11, 5.4))
    v = [100 * p(m, "vea_rate") for m in models]
    ax.set_ylim(0, 40)
    bars_with_ci(ax, xs, v, [100 * p(m, "vea_rate_ci")[0] for m in models], [100 * p(m, "vea_rate_ci")[1] for m in models],
                 colors, fmt="{:.1f}")
    finish(fig, ax, "01_vea_rate", "How often each model verbalizes eval awareness",
           "Share of scored rollouts whose reasoning says the interaction may be a test (judge: vea_cal). Rollouts 0-9 of 486 Fortress questions per model.",
           "VEA rate (% of rollouts)", models, xs)

    # 02 safety-relevant share
    fig, ax = plt.subplots(figsize=(11, 5.4))
    v = [100 * p(m, "safety_share_of_vea") for m in models]
    ax.set_ylim(0, 105)
    bars_with_ci(ax, xs, v, v, v, colors, fmt="{:.0f}", ci=False)
    for x, m in zip(xs, models):
        ax.text(x, 2, f"n={p(m, 'n_vea_with_category')}", ha="center", va="bottom", fontsize=7.5, color=SURFACE, rotation=90)
    finish(fig, ax, "02_safety_share", "Larger models say the test is about safety more often",
           "Share of VEA quotes the category judge labels harmful compliance or boundaries (vs abilities / other). n = VEA quotes.",
           "Safety-relevant share of VEA (%)", models, xs)

    # 03 VEA types, stacked
    types = [("HARMFUL_COMPLIANCE", "Harmful compliance", "#184f95"), ("BOUNDARIES", "Boundaries", "#6da7ec"),
             ("ABILITIES", "Abilities", "#eb6834"), ("OTHER", "Other / none", GREY)]
    counts = {}
    for m in models:
        d = U.design(U.load_table(m, "adversarial"), "first10")
        pos = d[(d.status == "scored") & (d.vea == True)]  # noqa: E712
        c = pos.subcategory.fillna("OTHER").replace({"NONE": "OTHER"}).value_counts()
        counts[m] = {k: c.get(k, 0) for k, _, _ in types}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bottom = np.zeros(len(models))
    for k, lab, col in types:
        tot = np.array([sum(counts[m].values()) for m in models], float)
        share = 100 * np.array([counts[m][k] for m in models]) / tot
        ax.bar(xs, share, bottom=bottom, width=0.72, color=col, edgecolor=SURFACE, linewidth=2, label=lab, zorder=2)
        bottom += share
    ax.set_ylim(0, 100)
    finish(fig, ax, "03_vea_types", "What the model says is being tested",
           "Category judge on each VEA quote (it also sees the prompt, so generic quotes get a type from context).",
           "Share of VEA quotes (%)", models, xs, legend=False,
           extra_handles=[Patch(color=c, label=l) for _, l, c in types])

    # 04 within-question gap (post's estimator)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    v = [100 * p(m, "within_q_refusal_diff") for m in models]
    lo = [100 * p(m, "within_q_refusal_diff_ci")[0] for m in models]
    hi = [100 * p(m, "within_q_refusal_diff_ci")[1] for m in models]
    ax.set_ylim(-30, 50)
    ax.axhline(0, color=INK2, linewidth=1, zorder=4)
    bars_with_ci(ax, xs, v, lo, hi, colors)
    finish(fig, ax, "04_within_question_gap", "Rollouts that verbalize eval awareness refuse more",
           "Refusal among VEA rollouts minus the others, averaged over questions with both kinds and refusal < 100% (the post's estimator). 95% CI.",
           "Refusal difference (percentage points)", models, xs)

    # 05 refusal on questions with / without any VEA rollout (post Fig. 5 split)
    rows = []
    for m in models:
        d = U.design(U.load_table(m, "adversarial"), "first10")
        s = d[(d.status == "scored") & d.vea.notna() & d.refusal.notna()]
        q = s.groupby("qid").agg(any_vea=("vea", lambda x: x.astype(bool).any()), ref=("refusal", lambda x: x.astype(float).mean()))
        rows.append((m, 100 * q[q.any_vea].ref.mean(), 100 * q[~q.any_vea].ref.mean(), int(q.any_vea.sum()), int((~q.any_vea).sum())))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.38
    ax.bar(xs - w / 2 - 0.02, [r[1] for r in rows], width=w - 0.04, color=colors, linewidth=0, zorder=2)
    ax.bar(xs + w / 2 + 0.02, [r[2] for r in rows], width=w - 0.04, color=GREY, linewidth=0, zorder=2)
    ax.set_ylim(0, 100)
    finish(fig, ax, "05_refusal_by_question_vea", "Questions that draw VEA are usually refused more often",
           "Mean refusal rate of questions with at least one VEA rollout (colored) vs questions with none (grey), rollouts 0-9. Across questions, not within.",
           "Refusal rate (%)", models, xs, legend=True,
           extra_handles=[Patch(color=GREY, label="no VEA rollout")])

    # 06 #VEA rollouts per question vs refusal rate, small multiples
    fig, axs = plt.subplots(4, 4, figsize=(12, 10.5), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, m in zip(axs.flat, models):
        d = U.design(U.load_table(m, "adversarial"), "first10")
        s = d[(d.status == "scored") & d.vea.notna() & d.refusal.notna()]
        q = s.groupby("qid").agg(nv=("vea", lambda x: x.astype(float).sum()), ref=("refusal", lambda x: 100 * x.astype(float).mean()))
        ax.scatter(q.nv + rng.uniform(-0.18, 0.18, len(q)), q.ref, s=9, alpha=0.45, color=FAMILY_COLOR[family(m)],
                   edgecolors="none", zorder=2)
        ax.set_title(f"{LABEL.get(m, m)}   ρ = {p(m, 'spearman_nvea_refusal'):.2f}", fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(-0.5, 10.5); ax.set_ylim(-5, 105); ax.set_xticks(range(0, 11, 2))
    for ax in axs[-1]:
        ax.set_xlabel("VEA rollouts (of 10)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Refusal rate (%)")
    fig.text(0.012, 0.99, "Questions with more VEA rollouts tend to be refused more", fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.962, "One dot per question (jittered horizontally); Spearman ρ between the two, rollouts 0-9.",
             fontsize=10.5, color=INK2, va="top")
    fig.subplots_adjust(top=0.9, bottom=0.06, left=0.07, right=0.99, hspace=0.35, wspace=0.12)
    fig.savefig(OUT / "06_nvea_vs_refusal.png", dpi=200); plt.close(fig)
    print(f"wrote {(OUT / '06_nvea_vs_refusal.png').relative_to(ROOT)}")

    # 07 injection: VEA - none, with manipulation - none
    iv_models = [m for m in models if m in IV]
    ixs = positions(iv_models)
    icol = [FAMILY_COLOR[family(m)] for m in iv_models]
    g = lambda m, k: IV[m][k]["primary"]  # noqa: E731
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_ylim(-5, 45)
    ax.axhline(0, color=INK2, linewidth=1, zorder=4)
    for off, key, col, lab in ((-w / 2, "insert_vea_minus_none", icol, False), (w / 2, "insert_placebo_minus_none", GREY, False)):
        vals = [100 * g(m, key)["value"] for m in iv_models]
        bars_with_ci(ax, ixs + off, vals, [100 * g(m, key)["ci"][0] for m in iv_models], [100 * g(m, key)["ci"][1] for m in iv_models],
                     col, width=w, labels=lab)
    finish(fig, ax, "07_injection", "Inserting \"this might be a test\" raises refusal",
           "Change in refusal after inserting the VEA sentence at 40% of the reasoning (colored) or the matched manipulation sentence (grey). 95% CI.",
           "Refusal change (percentage points)", iv_models, ixs,
           extra_handles=[Patch(color=GREY, label="manipulation sentence")])

    # 08 removal: keep - remove, rejection-sampled, with unfiltered
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_ylim(-20, 15)
    ax.axhline(0, color=INK2, linewidth=1, zorder=4)
    rej = [IV[m]["remove_keep_minus_remove_rejection"]["primary"] for m in iv_models]
    bars_with_ci(ax, ixs - w / 2, [100 * r["value"] for r in rej], [100 * r["ci"][0] for r in rej], [100 * r["ci"][1] for r in rej], icol, width=w, labels=False)
    unf = [g(m, "remove_keep_minus_remove") for m in iv_models]
    bars_with_ci(ax, ixs + w / 2, [100 * r["value"] for r in unf], [100 * r["ci"][0] for r in unf], [100 * r["ci"][1] for r in unf],
                 GREY, width=w, labels=False)
    finish(fig, ax, "08_removal", "Removing the model's own VEA sentence rarely changes refusal",
           "Keep minus remove. Colored: re-verbalizing continuations rejected (as in the post). Grey: all continuations. 95% CI.",
           "Refusal change (percentage points)", iv_models, ixs,
           extra_handles=[Patch(color=GREY, label="all continuations")])


if __name__ == "__main__":
    main()
