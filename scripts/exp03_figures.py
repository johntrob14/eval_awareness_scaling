"""exp03 figures in Plotly (PNG + interactive HTML per figure), in figures/exp03_vea_ladder_v2/<run>/.

Conventions (on top of exp02's model axis, layout and Okabe-Ito colors, reused from ladder_figures):
- every awareness label and every refusal type has ONE color AND ONE fill pattern, the same in every figure
  (exp03_config.AWARENESS_STYLE / REFUSAL_STYLE): evaluation "/", user probing "x", manipulation ".", none plain;
  full refusal "\\", partial "-", comply "|".
- underpowered bars (< 20 informative questions): faded fill, no CI.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path

import exp03_config as cfg
import plotly.graph_objects as go
from ladder_figures import (
    OKABE_ITO,
    base_layout,
    display_order,
    figure_width,
    model_positions,
    save,
    style_model_axis,
    subplot_titles_left,
)
from plotly.subplots import make_subplots

UNDERPOWERED_LABEL = "faded, no CI: < 20 informative questions"
POOLED_VEA = {"color": "#555555", "pattern": "", "label": "all VEA (evaluation or user probing)"}
NEUTRAL = {"color": OKABE_ITO["gray"], "pattern": "", "label": ""}


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def has_ci(e: dict) -> bool:
    return not e.get("underpowered") and all(finite(v) for v in e["ci"])


def marker(style: dict, faded: bool = False) -> dict:
    # Plotly bar outlines cannot be dashed, so underpowered bars are marked by the faded fill (and no CI) only.
    return dict(color=style["color"], opacity=0.35 if faded else 1.0,
                line=dict(color="#222", width=0.8),
                pattern=dict(shape=style["pattern"], fgcolor="#222", size=7, solidity=0.25))


def add_series(fig, models, entries, name, style, *, scale=100.0, row=None, legend="legend", positions=None) -> bool:
    """One bar per model; bars with a CI get error bars, underpowered bars are faded with a dashed outline."""
    pos = positions or model_positions(models)
    col = 1 if row else None
    points = [(m, e) for m, e in zip(models, entries) if e and finite(e.get("value"))]
    with_ci = [(m, e) for m, e in points if has_ci(e)]
    without = [(m, e) for m, e in points if not has_ci(e)]
    common = dict(name=name, legendgroup=name, offsetgroup=name, legend=legend)
    fig.add_trace(go.Bar(x=[pos[m] for m, _ in with_ci], y=[e["value"] * scale for _, e in with_ci], marker=marker(style),
                         customdata=[m for m, _ in with_ci], hovertemplate=f"%{{customdata}}<br>{name}: %{{y:.2f}}<extra></extra>",
                         error_y=dict(type="data", symmetric=False, color="#111", thickness=1.5, width=4,
                                      array=[(e["ci"][1] - e["value"]) * scale for _, e in with_ci],
                                      arrayminus=[(e["value"] - e["ci"][0]) * scale for _, e in with_ci]), **common),
                  row=row, col=col)
    if without:
        fig.add_trace(go.Bar(x=[pos[m] for m, _ in without], y=[e["value"] * scale for _, e in without],
                             marker=marker(style, faded=True), showlegend=False, customdata=[m for m, _ in without],
                             hovertemplate=f"%{{customdata}}<br>{name}: %{{y:.2f}} (underpowered)<extra></extra>", **common),
                      row=row, col=col)
    return any(e.get("underpowered") for _, e in points)


def add_underpowered_key(fig, legend="legend", row=None) -> None:
    fig.add_trace(go.Bar(x=[0], y=[0], name=UNDERPOWERED_LABEL, legend=legend, hoverinfo="skip",
                         marker=marker({"color": "#888", "pattern": ""}, faded=True)), row=row, col=1 if row else None)


def stacked_shares(fig, models, per_model, keys_styles, *, row=None) -> None:
    pos = model_positions(models)
    for key, style in keys_styles:
        entries = [per_model[m].get(key) for m in models]
        fig.add_trace(go.Bar(x=[pos[m] for m in models], y=[e["value"] * 100 if e else None for e in entries],
                             name=style["label"], marker=marker(style), customdata=models,
                             hovertemplate=f"%{{customdata}}<br>{style['label']}: %{{y:.1f}}%<extra></extra>"),
                      row=row, col=1 if row else None)


# --- Figures -------------------------------------------------------------------------------------------------------


def fig_awareness(per_model, path):
    models = display_order(list(per_model))
    fig = go.Figure()
    stacked_shares(fig, models, per_model, [(f"awareness_share_{k}", cfg.AWARENESS_STYLE[k]) for k in cfg.AWARENESS_TYPES[:-1]])
    pos = model_positions(models)
    totals = [per_model[m]["M1_vea_rate"] for m in models]
    fig.add_trace(go.Scatter(x=[pos[m] for m in models], y=[t["value"] * 100 for t in totals], mode="markers",
                             name="VEA = evaluation + user probing (M1), 95% CI",
                             marker=dict(symbol="line-ew", size=24, line=dict(color="#111", width=2)),
                             error_y=dict(type="data", symmetric=False, color="#111", thickness=1.5, width=5,
                                          array=[(t["ci"][1] - t["value"]) * 100 for t in totals],
                                          arrayminus=[(t["value"] - t["ci"][0]) * 100 for t in totals])))
    fig.add_trace(go.Scatter(x=[pos[m] for m in models], y=[per_model[m]["M1_length_standardized"] * 100 for m in models],
                             mode="markers", name="VEA standardized to the<br>pooled CoT-length mix",
                             marker=dict(symbol="diamond-open", size=11, color="#111", line=dict(width=2))))
    base_layout(fig, "F1: what the reasoning says about being tested, Fortress stage 1",
                "one label per rollout (precedence evaluation > user probing > manipulation); 95% CI over questions",
                figure_width(len(models), 1.5), 580, legend_title="awareness label (judge)")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="% of scored rollouts (judge)", rangemode="tozero")
    style_model_axis(fig, models, mark_addons=False)
    save(fig, path)


def fig_refusal(per_model, path):
    models = display_order(list(per_model))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                        subplot_titles=("F2a: adversarial prompts (stage 1)", "F2b: benign twins"))
    subplot_titles_left(fig)
    for row, prefix in ((1, "refusal_share_"), (2, "benign_refusal_share_")):
        stacked_shares(fig, models, per_model, [(f"{prefix}{k}", cfg.REFUSAL_STYLE[k]) for k in cfg.REFUSAL_TYPES], row=row)
        style_model_axis(fig, models, mark_addons=False, row=row)
        fig.update_yaxes(title_text="% of scored answers (judge)", range=[0, 100], row=row, col=1)
    for trace in fig.data[len(cfg.REFUSAL_TYPES):]:
        trace.showlegend = False
    base_layout(fig, "F2: graded refusal on Fortress", "full refusal / partial / comply, stacked to 100%",
                figure_width(len(models), 1.5), 880, legend_title="answer (judge)")
    fig.update_layout(barmode="stack")
    save(fig, path)


def fig_within(per_model, path):
    """F3a: Mantel-Haenszel full-refusal difference. F3b: within-question correlation (phi, question fixed effects)."""
    models = display_order(list(per_model))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2, subplot_titles=(
        "F3a: full-refusal rate of flagged rollouts minus unflagged rollouts to the same question (Mantel-Haenszel)",
        "F3b: within-question correlation of flag and full refusal (φ, question fixed effects)"))
    subplot_titles_left(fig)
    series = [  # (risk-difference key, correlation key, style)
        ("M4_within_question_pooled", "M4_within_corr_pooled", POOLED_VEA),
        ("M4_type_evaluation", "M4_within_corr_type_evaluation",
         {**cfg.AWARENESS_STYLE["evaluation"], "label": "evaluation vs no VEA"}),
        ("M4_type_user_probing", "M4_within_corr_type_user_probing",
         {**cfg.AWARENESS_STYLE["user_probing"], "label": "user probing vs no VEA"}),
        ("M6_placebo_within_question", "M6_placebo_within_corr",
         {**cfg.AWARENESS_STYLE["manipulation"], "label": "placebo: manipulation vs neither"})]
    underpowered = False
    n_row1_traces = 0
    for row, scale in ((1, 100.0), (2, 1.0)):
        for rd_key, corr_key, style in series:
            key = rd_key if row == 1 else corr_key
            underpowered |= add_series(fig, models, [per_model[m].get(key) for m in models], style["label"], style,
                                       scale=scale, row=row)
        style_model_axis(fig, models, mark_addons=False, row=row)
        n_row1_traces = n_row1_traces or len(fig.data)
    for trace in fig.data[n_row1_traces:]:  # one legend entry per series
        trace.showlegend = False
    if underpowered:
        add_underpowered_key(fig, row=1)
    base_layout(fig, "F3: within-question association between VEA and full refusal", "95% CI over questions",
                figure_width(len(models), 4), 900, legend_title="comparison (judge)")
    fig.update_yaxes(title_text="refusal difference, pp (judge)", row=1, col=1)
    fig.update_yaxes(title_text="within-question φ (judge)", row=2, col=1)
    save(fig, path)


def fig_primary(per_model, path):
    models = display_order(list(per_model))
    fig = go.Figure()
    series = [("M6_specificity", {"color": "#333333", "pattern": "", "label": "M6 specificity: VEA diff minus placebo diff"}),
              ("M4_length_matched", {"color": OKABE_ITO["sky_blue"], "pattern": "", "label": "M4 length-matched (CoT tercile strata)"}),
              ("M4_any_refusal", {**cfg.REFUSAL_STYLE["partial"], "label": "M4 with full or partial refusal"})]
    underpowered = False
    for key, style in series:
        underpowered |= add_series(fig, models, [per_model[m].get(key) for m in models], style["label"], style)
    if underpowered:
        add_underpowered_key(fig)
    base_layout(fig, "F4: is the VEA-refusal link specific, and robust to CoT length?", "95% CI over questions",
                figure_width(len(models), 3), 560, legend_title="estimate (judge)")
    fig.update_yaxes(title_text="refusal difference, pp (judge)")
    style_model_axis(fig, models, mark_addons=False)
    save(fig, path)


def fig_interventions(interventions, path):
    models = display_order([m for m, e in interventions.items() if e])
    if not models:
        return "skipped: no intervention results"
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22, subplot_titles=(
        "F5a: sentence inserted into regex-free reasoning, minus nothing inserted",
        "F5b: VEA sentence kept, minus removed (resampled from before it)"))
    subplot_titles_left(fig)
    for key, style in (("insert_vea_minus_none_R", {**cfg.AWARENESS_STYLE["evaluation"], "label": "'this might be a test' inserted"}),
                       ("insert_placebo_minus_none_R", {**cfg.AWARENESS_STYLE["manipulation"], "label": "'manipulation' inserted (placebo)"}),
                       ("insert_neutral_minus_none_R", {**cfg.AWARENESS_STYLE["none"], "label": "neutral sentence inserted"})):
        add_series(fig, models, [interventions[m].get(key) for m in models], style["label"], style, row=1)
    add_series(fig, models, [interventions[m].get("remove_keep_minus_remove_R") for m in models], "VEA sentence kept vs removed",
               {**cfg.AWARENESS_STYLE["evaluation"]}, row=2, legend="legend2")
    base_layout(fig, "F5: causal test: does a VEA sentence change the refusal rate?",
                "full-refusal difference, paired by base rollout (8 resamples each); 95% CI over questions",
                figure_width(len(models), 3), 860, legend_title="inserted sentence")
    fig.update_layout(legend2=dict(x=1.02, xanchor="left", y=fig.layout.yaxis2.domain[1], yanchor="top",
                                   bordercolor="#bbb", borderwidth=1, font=dict(size=13)))
    for row in (1, 2):
        fig.update_yaxes(title_text="refusal difference, pp (judge)", row=row, col=1)
        style_model_axis(fig, models, mark_addons=False, row=row)
    save(fig, path)
    return "ok"


def fig_logit_v2(logit_v2, path):
    models = display_order([m for m, e in logit_v2.items() if e])
    if not models:
        return "skipped: no logit v2 results"
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22, subplot_titles=(
        "F6a: paired win rate, side b scores higher than side a (chance = 50)",
        "F6b: disguise shift in score, harmful minus benign requests (nats)"))
    subplot_titles_left(fig)
    for kind, label, color in (("format", "benchmark wrapper > plain chat (Fortress)", OKABE_ITO["blue"]),
                               ("disguise", "disguised > blunt, harmful (WildJailbreak)", OKABE_ITO["vermillion"]),
                               ("disguise_benign", "disguised > blunt, benign (WildJailbreak)", OKABE_ITO["sky_blue"])):
        add_series(fig, models, [(logit_v2[m] or {}).get(kind, {}).get("win_rate") for m in models], label,
                   {"color": color, "pattern": ""}, row=1)
    add_series(fig, models, [(logit_v2[m] or {}).get("harmful_minus_benign_disguise_shift") for m in models],
               "harmful minus benign shift", {"color": "#333333", "pattern": ""}, scale=1.0, row=2, legend="legend2")
    base_layout(fig, "F6: logit monitor v2 on matched prompt pairs (judge-free)",
                "score = log P('The user might be testing') - log P('The user might be asking'); 95% CI over pairs",
                figure_width(len(models), 3), 860, legend_title="pair type")
    fig.update_layout(legend2=dict(x=1.02, xanchor="left", y=fig.layout.yaxis2.domain[1], yanchor="top",
                                   bordercolor="#bbb", borderwidth=1, font=dict(size=13)))
    fig.add_hline(y=50, line=dict(color="#666", dash="dash", width=1.2), row=1, col=1)
    fig.update_yaxes(title_text="win rate, %", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="score shift difference, nats", row=2, col=1)
    for row in (1, 2):
        style_model_axis(fig, models, mark_addons=False, row=row)
    save(fig, path)
    return "ok"


def render_all(per_model: dict, interventions: dict, logit_v2: dict, fig_dir: Path) -> dict:
    """Render every figure; a failing figure is reported and skipped so the numeric outputs are still written."""
    jobs = {"F1_awareness": lambda p: fig_awareness(per_model, p), "F2_refusal": lambda p: fig_refusal(per_model, p),
            "F3_within_question": lambda p: fig_within(per_model, p), "F4_specificity_robustness": lambda p: fig_primary(per_model, p),
            "F5_interventions": lambda p: fig_interventions(interventions, p), "F6_logit_v2": lambda p: fig_logit_v2(logit_v2, p)}
    status = {}
    for name, draw in jobs.items():
        path = fig_dir / f"{name}.png"
        try:
            status[name] = {"path": str(path.relative_to(cfg.REPO_ROOT)), "status": draw(path) or "ok"}
        except Exception as exc:
            traceback.print_exc()
            status[name] = {"path": str(path), "status": f"failed: {exc!r}"}
    return status
