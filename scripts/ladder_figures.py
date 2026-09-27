"""exp02 figures F1-F5 in Plotly: a PNG plus an interactive HTML (same stem) per figure.

ladder_analysis.py calls render_all() with in-memory metrics; each analysis run gets its own folder,
figures/exp02_vea_ladder/<out-name>/. To re-render from a summary.json (run in /venv/main):
    python scripts/ladder_figures.py results/exp02_vea_ladder/summary.json --out-dir figures/exp02_vea_ladder/rerender
    python scripts/ladder_figures.py results/exp02_vea_ladder/summary.json --replace   # the recorded paths

Conventions shared by every figure:
- x axis: models grouped by family (Qwen3 / Qwen3.5 / Qwen3.8) and ordered by size. Color never encodes family.
- color: one distinct Okabe-Ito hue (colorblind-safe) per series; a framing has the same color in every figure.
- striped bars without a CI: fewer than 20 informative questions (underpowered).
- legends sit outside the plot area (right of it), so they never cover bars.
PNG export needs Chrome (installed with `plotly_get_chrome`); the HTML is written first, so it exists even if PNG export fails.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ladder_config as cfg

OKABE_ITO = {"orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
             "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999"}
FRAMING_COLORS = {"safety": OKABE_ITO["vermillion"], "capabilities": OKABE_ITO["blue"],
                  "both": OKABE_ITO["green"], "generic": OKABE_ITO["purple"]}
FAMILY_LABELS = {"qwen3": "Qwen3", "qwen3.5": "Qwen3.5", "qwen3.8": "Qwen3.8"}
CONTRAST_LABELS = {"toxicchat_toxic": "vs ToxicChat toxic", "wildchat_benign": "vs WildChat benign",
                   "toxicchat_benign": "vs ToxicChat benign", "fortress_benign": "vs Fortress benign twins"}
UNDERPOWERED_LABEL = "striped, no CI: < 20 informative questions"
FONT = dict(family="Arial, Helvetica, sans-serif", size=15, color="#222")
LEGEND_PX = 350  # room reserved right of the plot for the widest legend
MIN_WIDTH = 950  # keeps the longest (sub)titles unclipped with few models
FAMILY_GAP = 0.6  # extra x space between model families


# --- Model axis ---------------------------------------------------------------------------------

def family(model: str) -> str:
    return cfg.ALL_SUBJECTS[model]["family"]


def display_order(models: list[str]) -> list[str]:
    families = list(FAMILY_LABELS)
    return sorted(models, key=lambda m: (families.index(family(m)), cfg.ALL_SUBJECTS[m]["params_b"]))


def model_positions(models: list[str]) -> dict[str, float]:
    """x position per model (already in display order): 1 apart, plus FAMILY_GAP between families."""
    pos, x = {}, 0.0
    for prev, m in zip([None] + models[:-1], models):
        if prev is not None:
            x += 1 + (FAMILY_GAP if family(m) != family(prev) else 0)
        pos[m] = x
    return pos


def style_model_axis(fig, models: list[str], mark_addons: bool, row: int | None = None) -> None:
    """Size tick labels, family names under each group, and a light separator line between families.
    row: panel of a make_subplots figure (single column); None for a single-panel figure."""
    pos = model_positions(models)
    suffix = {m: "<br>(stage 1 only)" if mark_addons and m in cfg.ADDON_SUBJECTS else "" for m in models}
    xref, yref = ("x", "y") if row in (None, 1) else (f"x{row}", f"y{row}")
    fig.update_xaxes(tickvals=[pos[m] for m in models], ticktext=[m.split("-", 1)[1] + suffix[m] for m in models],
                     range=[-0.6, pos[models[-1]] + 0.6], tickangle=0, showgrid=False, row=row, col=1 if row else None)
    groups = [[m for m in models if family(m) == f] for f in FAMILY_LABELS]
    groups = [g for g in groups if g]
    two_line_ticks = any(suffix.values())
    for g in groups:
        fig.add_annotation(x=(pos[g[0]] + pos[g[-1]]) / 2, y=0, xref=xref, yref=f"{yref} domain", yanchor="top",
                           yshift=-50 if two_line_ticks else -34, text=f"<b>{FAMILY_LABELS[family(g[0])]}</b>", showarrow=False, font=dict(size=15))
    for left, right in zip(groups[:-1], groups[1:]):
        x = (pos[left[-1]] + pos[right[0]]) / 2
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref=xref, yref=f"{yref} domain",
                      line=dict(color="#aaa", width=1, dash="dot"))


# --- Bars -----------------------------------------------------------------------------------------

def finite(x) -> bool:
    return x is not None and math.isfinite(x)


def has_ci(entry: dict) -> bool:
    return not entry.get("underpowered") and all(finite(v) for v in entry["ci"])


def add_bar_series(fig, models, entries, name, color, *, scale=100.0, row=None, legend="legend",
                   showlegend=True) -> bool:
    """One bar per model for one series. Bars with a CI get error bars; bars without one (underpowered,
    or undefined CI) are drawn as stripes in the series color. Returns True if any bar was underpowered."""
    pos = model_positions(models)
    col = 1 if row else None
    points = [(m, e) for m, e in zip(models, entries) if e is not None and finite(e["value"])]

    def bar(subset, **kwargs):
        hover = [[*(v * scale if finite(v) else float("nan") for v in e["ci"]), m] for m, e in subset]
        return go.Bar(x=[pos[m] for m, _ in subset], y=[e["value"] * scale for _, e in subset],
                      name=name, legendgroup=name, offsetgroup=name, legend=legend, customdata=hover,
                      hovertemplate=f"%{{customdata[2]}}<br>{name}: %{{y:.2f}} "
                                    f"[%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]<extra></extra>",
                      **kwargs)

    with_ci = [(m, e) for m, e in points if has_ci(e)]
    fig.add_trace(bar(with_ci, showlegend=showlegend,
                      marker=dict(color=color, line=dict(color="#222", width=0.8)),
                      error_y=dict(type="data", symmetric=False, color="#111", thickness=1.5, width=4,
                                   array=[(e["ci"][1] - e["value"]) * scale for _, e in with_ci],
                                   arrayminus=[(e["value"] - e["ci"][0]) * scale for _, e in with_ci])),
                  row=row, col=col)
    without_ci = [(m, e) for m, e in points if not has_ci(e)]
    if without_ci:
        fig.add_trace(bar(without_ci, showlegend=False,
                          marker=dict(color=color, line=dict(color=color, width=1.5),
                                      pattern=dict(shape="/", fillmode="replace", size=7, solidity=0.45))),
                      row=row, col=col)
    return any(e.get("underpowered") for _, e in points)


def add_underpowered_key(fig, first_series, *, legend="legend", row=None) -> None:
    """Legend-only entry explaining striped bars (a zero-height bar in the first series' slot)."""
    fig.add_trace(go.Bar(x=[0], y=[0], name=UNDERPOWERED_LABEL, offsetgroup=first_series, legend=legend,
                         hoverinfo="skip", marker=dict(color="#666", line=dict(color="#666", width=1.5),
                                                        pattern=dict(shape="/", fillmode="replace", size=7, solidity=0.45))),
                  row=row, col=1 if row else None)


def grouped_series(fig, models, series, *, row=None, legend="legend", scale=100.0, showlegend=True) -> None:
    """series: list of (name, color, entries); adds the striped-bar key when any bar is underpowered."""
    underpowered = False
    for name, color, entries in series:
        underpowered |= add_bar_series(fig, models, entries, name, color, scale=scale, row=row, legend=legend,
                                       showlegend=showlegend)
    if underpowered and showlegend:
        add_underpowered_key(fig, series[0][0], legend=legend, row=row)


# --- Layout -------------------------------------------------------------------------------------

def base_layout(fig, title: str, subtitle: str, width: int, height: int, legend_title: str | None = None) -> None:
    fig.update_layout(
        template="plotly_white", font=FONT, width=width, height=height, barmode="group", bargap=0.22, bargroupgap=0.06,
        title=dict(text=f"<b>{title}</b>", x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(size=18),
                   subtitle=dict(text=subtitle, font=dict(size=13, color="#555"))),
        legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top", title=dict(text=legend_title),
                    bordercolor="#bbb", borderwidth=1, font=dict(size=13), tracegroupgap=4),
        margin=dict(l=90, r=30, t=110, b=95),
    )
    fig.update_xaxes(showline=True, linecolor="#222", tickfont=dict(size=14))
    fig.update_yaxes(showline=True, linecolor="#222", gridcolor="#e3e3e3", zeroline=True, zerolinecolor="#222",
                     zerolinewidth=1.2, tickfont=dict(size=14), title_font=dict(size=15))


def figure_width(n_models: int, n_series: float, legend: bool = True) -> int:
    return max(MIN_WIDTH, int(110 + (LEGEND_PX if legend else 60) + n_models * (34 + 14 * n_series)))


def save(fig, path: Path) -> None:
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(path, scale=2)


# --- F1-F5 --------------------------------------------------------------------------------------

def subplot_titles_left(fig) -> None:
    """Left-align make_subplots' panel titles; call before adding any other annotation."""
    fig.update_annotations(font=dict(size=15), x=0, xanchor="left")


def fig_vea(metrics, models, path):
    models = display_order(models)
    pos = model_positions(models)
    x = [pos[m] for m in models]
    fig = go.Figure()
    for f, color in FRAMING_COLORS.items():
        fig.add_trace(go.Bar(x=x, y=[metrics[m][f"M2_rate_{f}"]["value"] * 100 for m in models], name=f,
                             marker=dict(color=color, line=dict(color="#222", width=0.6)), customdata=models,
                             hovertemplate=f"%{{customdata}}<br>{f}: %{{y:.2f}}%<extra></extra>"))
    totals = [metrics[m]["M1_vea_rate"] for m in models]
    fig.add_trace(go.Scatter(
        x=x, y=[t["value"] * 100 for t in totals], mode="markers", name="all framings (M1), 95% CI",
        marker=dict(symbol="line-ew", size=24, line=dict(color="#111", width=2)), customdata=models,
        error_y=dict(type="data", symmetric=False, color="#111", thickness=1.5, width=5,
                     array=[(t["ci"][1] - t["value"]) * 100 for t in totals],
                     arrayminus=[(t["value"] - t["ci"][0]) * 100 for t in totals]),
        hovertemplate="%{customdata}<br>all framings: %{y:.2f}%<extra></extra>"))
    bloom = [m for m in models if m in cfg.BLOOM_A1_QWEN3_VEA]
    if bloom:
        fig.add_trace(go.Scatter(x=[pos[m] for m in bloom], y=[cfg.BLOOM_A1_QWEN3_VEA[m] * 100 for m in bloom],
                                 mode="markers", name="Bloom et al. A1<br>(GPT-5-mini judge)", customdata=bloom,
                                 marker=dict(symbol="diamond", size=11, color="white", line=dict(color="#111", width=2)),
                                 hovertemplate="%{customdata}<br>Bloom A1: %{y:.1f}%<extra></extra>"))
    base_layout(fig, "F1: verbalized eval awareness (VEA) on Fortress, stage 1",
                "stacked by what the model thinks is being evaluated; 95% CI over questions",
                figure_width(len(models), 1.5), 560, legend_title="VEA framing")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="VEA rate, % of scored rollouts (judge)", rangemode="tozero")
    style_model_axis(fig, models, mark_addons=True)
    save(fig, path)


def fig_refusal(metrics, models, path):
    models = display_order(models)
    fig = go.Figure()
    grouped_series(fig, models, [
        ("adversarial prompt (stage 1)", OKABE_ITO["vermillion"], [metrics[m]["M3_refusal_adversarial"] for m in models]),
        ("benign twin", OKABE_ITO["sky_blue"], [metrics[m]["M3_refusal_benign"] for m in models]),
    ])
    base_layout(fig, "F2: refusal on Fortress adversarial prompts vs their benign twins", "95% CI over questions",
                figure_width(len(models), 2), 540)
    fig.update_yaxes(title_text="refusal rate, % (judge)", rangemode="tozero")
    style_model_axis(fig, models, mark_addons=True)
    save(fig, path)


WITHIN_SERIES = [  # (legend name, color, risk-difference key, within-question correlation key)
    ("pooled: VEA vs non-VEA", OKABE_ITO["gray"], "M4_within_question_pooled", "M4_within_corr_pooled"),
    ("safety-framed VEA vs non-VEA", FRAMING_COLORS["safety"], "M4_framing_safety", "M4_within_corr_framing_safety"),
    ("capabilities-framed VEA vs non-VEA", FRAMING_COLORS["capabilities"], "M4_framing_capabilities",
     "M4_within_corr_framing_capabilities"),
    ("placebo: jailbreak-noticing vs neither", OKABE_ITO["orange"], "M6_placebo_within_question", "M6_placebo_within_corr"),
]


def fig_within(metrics, models, path):
    """F3a: Mantel-Haenszel risk difference. F3b (summaries that have it): within-question correlation."""
    models = display_order(models)
    with_corr = all(corr_key in metrics[m] for m in models for *_, corr_key in WITHIN_SERIES)
    rows = 2 if with_corr else 1
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.24, subplot_titles=(
        "F3a: refusal rate of flagged rollouts minus other rollouts to the same question (Mantel-Haenszel)",
        "F3b: within-question correlation of flag and refusal (φ, question fixed effects)")[:rows])
    subplot_titles_left(fig)
    grouped_series(fig, models, [(name, color, [metrics[m][rd] for m in models])
                                 for name, color, rd, _ in WITHIN_SERIES], row=1)
    if with_corr:
        grouped_series(fig, models, [(name, color, [metrics[m][corr] for m in models])
                                     for name, color, _, corr in WITHIN_SERIES], row=2, scale=1.0, showlegend=False)
    base_layout(fig, "F3: within-question association between VEA and refusal", "95% CI over questions",
                figure_width(len(models), 4), 560 if rows == 1 else 900, legend_title="comparison")
    fig.update_yaxes(title_text="refusal difference, pp (judge)", row=1, col=1)
    if with_corr:
        fig.update_yaxes(title_text="within-question φ (judge)", row=2, col=1)
    for row in range(1, rows + 1):
        style_model_axis(fig, models, mark_addons=True, row=row)
    save(fig, path)


def fig_across(metrics, models, path):
    models = display_order(models)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.24, subplot_titles=(
        "F4a: non-VEA refusal, questions with ≥ 1 VEA rollout (of 10) minus questions with none",
        "F4b: Spearman ρ across questions, VEA share vs non-VEA refusal"))
    subplot_titles_left(fig)
    for row, key, scale, color in ((1, "M5a_gap_nonvea_refusal", 100.0, OKABE_ITO["blue"]),
                                   (2, "M5b_spearman_v_r0", 1.0, OKABE_ITO["green"])):
        add_bar_series(fig, models, [metrics[m][key] for m in models], key, color, scale=scale, row=row, showlegend=False)
        style_model_axis(fig, models, mark_addons=True, row=row)
    base_layout(fig, "F4: across-question association between VEA and refusal", "95% CI over questions",
                figure_width(len(models), 1.5, legend=False), 820)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="refusal gap, pp (judge)", row=1, col=1)
    fig.update_yaxes(title_text="Spearman ρ (judge)", row=2, col=1)
    save(fig, path)


def fig_auroc(metrics, models, path):
    first = cfg.LOGIT_MONITOR_CONTRASTS[0]
    primary = f"M7_auroc_vs_{first}"
    models = display_order([m for m in models if primary in metrics[m]])
    if not models:
        return
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22, subplot_titles=(
        f"F5a: each score, Fortress adversarial {CONTRAST_LABELS.get(first, f'vs {first}')}",
        "F5b: raw score, Fortress adversarial vs each comparison set"))
    subplot_titles_left(fig)
    scores = [("primary", "raw score", OKABE_ITO["blue"]), ("contrast", "contrast score", OKABE_ITO["green"]),
              ("primary_length_residualized", "raw, length-residualized", OKABE_ITO["purple"]),
              ("length_only_baseline", "prompt length only (baseline)", OKABE_ITO["gray"])]
    grouped_series(fig, models, [(label, color, [metrics[m][primary][k] for m in models]) for k, label, color in scores],
                   row=1, legend="legend")
    contrast_colors = [OKABE_ITO["vermillion"], OKABE_ITO["sky_blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    grouped_series(fig, models, [(CONTRAST_LABELS.get(o, f"vs {o}"), color,
                                  [metrics[m].get(f"M7_auroc_vs_{o}", {}).get("primary") for m in models])
                                 for o, color in zip(cfg.LOGIT_MONITOR_CONTRASTS, contrast_colors)],
                   row=2, legend="legend2")
    base_layout(fig, "F5: logit monitor AUROC (judge-free)", "chance = 50 (dashed line); 95% CI over prompts",
                figure_width(len(models), 4), 900, legend_title="score")
    fig.update_layout(legend2=dict(x=1.02, xanchor="left", y=fig.layout.yaxis2.domain[1], yanchor="top",
                                   title=dict(text="comparison set"), bordercolor="#bbb", borderwidth=1, font=dict(size=13),
                                   tracegroupgap=4))
    for row in (1, 2):
        fig.add_hline(y=50, line=dict(color="#666", dash="dash", width=1.2), row=row, col=1)
        fig.update_yaxes(title_text="AUROC × 100", range=[0, 100], row=row, col=1)
        style_model_axis(fig, models, mark_addons=False, row=row)
    save(fig, path)


def render_all(metrics: dict, models: list[str], paths: dict[str, Path], logit_only: dict | None = None) -> dict[str, str]:
    """Render F1-F5. A failing figure is reported and skipped, so the caller still writes its numeric outputs."""
    logit_only = logit_only or {}
    auroc_models = [m for m in cfg.RUN_ORDER if m in metrics or m in logit_only]
    jobs = {"F1": (fig_vea, metrics, models), "F2": (fig_refusal, metrics, models), "F3": (fig_within, metrics, models),
            "F4": (fig_across, metrics, models), "F5": (fig_auroc, {**metrics, **logit_only}, auroc_models)}
    status = {}
    for key, (draw, data, ms) in jobs.items():
        try:
            draw(data, ms, paths[key])
            status[key] = "ok"
        except Exception as exc:
            traceback.print_exc()
            status[key] = f"failed: {exc!r}"
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render F1-F5 from a ladder_analysis summary.json.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out-dir", type=Path, help="write here instead of the paths recorded in the summary")
    parser.add_argument("--replace", action="store_true", help="allow replacing existing figure files")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    paths = {k: cfg.REPO_ROOT / v for k, v in summary["figures"].items()}
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        paths = {k: args.out_dir / p.name for k, p in paths.items()}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not args.replace:
        raise FileExistsError(f"{existing} exist; pass --replace to re-render them")
    print(render_all(summary["per_model"], summary["models"], paths, summary.get("logit_monitor_only", {})))


if __name__ == "__main__":
    main()
