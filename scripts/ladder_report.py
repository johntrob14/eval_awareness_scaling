"""exp02: draft the AGENTS.md-format report from summary.json (numbers only; no interpretation).

    python scripts/ladder_report.py                       # results/exp02_vea_ladder/REPORT_auto.md
    python scripts/ladder_report.py --out-name interim_T20

Written by the unattended tail so a complete, checkable report exists even if nobody is around;
the agent's (or a human's) interpretation goes in a separate REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import ladder_config as cfg


def fmt(entry: dict | None, unit: str = "%") -> str:
    """Estimate with its CI; unit '%' (rates), 'pp' (differences) or '' (AUROC, unscaled)."""
    if not entry or entry.get("value") is None:
        return "n/a"
    scale = 1 if unit == "" else 100
    digits = 2 if unit == "" else 1
    lo, hi = entry.get("ci") or [None, None]
    ci = f" [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]" if lo is not None and hi is not None else " [CI n/a]"
    flag = " (underpowered)" if entry.get("underpowered") else ""
    return f"{entry['value'] * scale:.{digits}f}{unit}{ci}{flag}"


def fmt_diff(step: dict) -> str:
    """Difference between two models in percentage points, with its paired-bootstrap CI."""
    if step["diff"] is None:
        return "n/a"
    lo, hi = step["ci"]
    ci = "" if lo is None else f" [{lo * 100:+.1f}, {hi * 100:+.1f}]"
    return f"{step['diff'] * 100:+.1f}{ci}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", default="final")
    args = parser.parse_args()
    out_dir = cfg.RESULTS_DIR if args.out_name == "final" else cfg.RESULTS_DIR / args.out_name
    s = json.loads((out_dir / "summary.json").read_text())
    manifest = json.loads(cfg.MANIFEST_PATH.read_text())
    commit = subprocess.run(["git", "-C", str(cfg.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    models = s["models"]
    pm = s["per_model"]
    primary_contrast = f"M7_auroc_vs_{cfg.LOGIT_MONITOR_CONTRASTS[0]}"

    lines = [
        f"# {cfg.EXP_ID} report ({args.out_name}), auto-drafted from summary.json",
        "",
        f"EXP {cfg.EXP_ID} @ {commit} | model {', '.join(models)} | n=489 Fortress adversarial questions "
        f"(10 stage-1 rollouts each; +{cfg.STAGE2_EXTRA_USED} on the 98 selected questions) | "
        f"seeds=per-rollout seeds from rollout_seed(prompt_set, item, index); engine {cfg.VLLM_ENGINE_SEED}; "
        f"bootstrap {cfg.BOOTSTRAP_SEED}",
        "",
        "Claim tested: within the same Fortress question, rollouts whose CoT verbalizes eval awareness (VEA) "
        "refuse more often than rollouts that do not (M4, pre-registered primary with M1), and whether VEA "
        "rate or that effect jumps between adjacent sizes within a family.",
        "",
        "Evidence grade: every VEA / refusal / placebo number below is **judge**-labeled (Qwen3.8-27B bf16, "
        "non-thinking); hand-label agreement is pending unless listed under judge checks. The logit-monitor "
        "AUROCs are judge-free.",
        "",
        "## Headline per model (95% question-bootstrap CI)",
        "",
        "| model | VEA rate M1 (judge) | safety-framed VEA share | refusal adv / benign (judge) | "
        "within-question M4 (judge) | M4 stage-2 only | placebo M6 (judge) | positive control VEA | "
        "logit AUROC raw / length-only |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        e = pm[m]
        share = e["M2_framing_share"]
        safety = share.get("safety")
        safety_share = "n/a" if safety is None else f"{(safety + (share.get('both') or 0)) * 100:.0f}%"
        auroc = e.get(primary_contrast, {})
        pos = e["positive_control_eval_stating_vea_rate"]
        positive_control = "n/a" if pos["value"] is None else f"{pos['value'] * 100:.0f}% (n={pos['n']})"
        m4 = e["M4_within_question_pooled"]
        name = m + (" (stage 1 only)" if m in cfg.ADDON_SUBJECTS else "")
        lines.append(
            f"| {name} | {fmt(e['M1_vea_rate'])} | {safety_share} | "
            f"{fmt(e['M3_refusal_adversarial'])} / {fmt(e['M3_refusal_benign'])} | "
            f"{fmt(m4, 'pp')} (n_inf {m4.get('n_informative_questions')}) | "
            f"{fmt(e.get('M4_selection_free_stage2_only'), 'pp')} | {fmt(e['M6_placebo_within_question'], 'pp')} | "
            f"{positive_control} | "
            f"{fmt(auroc.get('primary'), unit='')} / {fmt(auroc.get('length_only_baseline'), unit='')} |")
    for m, e in s.get("logit_monitor_only", {}).items():
        auroc = e.get(primary_contrast, {})
        lines.append(f"| {m} (logit monitor only) | | | | | | | | "
                     f"{fmt(auroc.get('primary'), unit='')} / {fmt(auroc.get('length_only_baseline'), unit='')} |")

    lines += ["", "Baseline for M4: the placebo contrast M6 (jailbreak-noticing without VEA, same estimator). "
              "Control: the 11 eval-stating prompts (VEA should be high). Bloom et al. A1 (GPT-5-mini judge) for Qwen3:",
              ""]
    for m, v in s.get("judge_swap_vs_bloom_a1", {}).items():
        lines.append(f"- {m}: ours {v['ours'] * 100:.1f}% [{v['ours_ci'][0] * 100:.1f}, {v['ours_ci'][1] * 100:.1f}] vs Bloom {v['bloom_a1'] * 100:.1f}%")

    lines += ["", "## Jump tests (manifest rule; primary outcomes only)", ""]
    for key, j in s["jump_tests"].items():
        if "steps" in j:
            steps = "; ".join(f"{st['from']}->{st['to']} {fmt_diff(st)}{' u/p' if st['underpowered'] else ''}"
                              for st in j["steps"])
            note = " (involves an underpowered model)" if j.get("jump_involves_underpowered_model") else ""
            lines.append(f"- {key}: jump = {j['jump']}{note}. Steps (pp): {steps}")
        else:
            lines.append(f"- {key} (exploratory): {j['from']} -> {j['to']} {fmt_diff(j)} pp")

    checks = s["judge_checks"]
    lines += ["", "## Judge checks (M8)", "", "```", json.dumps(checks, indent=1)[:6000], "```"]

    lines += [
        "", "## Figures", "",
        *[f"- {k}: {v}" for k, v in s["figures"].items()],
        "",
        "## Three dumbest ways this could be wrong",
        "",
        "1) The judge over-counts VEA (the Bloom rubric counts 'the user might be testing my boundaries'; Qwen3.8-27B "
        "judges itself). Checked by: self-agreement, window vs full-CoT agreement, thinking-mode and FP8 re-judging "
        "(judge checks above), the positive control, and Bloom A1 rates. Not yet checked: hand labels (sheets in "
        "labeling/).",
        "2) The within-question gap comes from the adaptive design or from length, not VEA: stage-2 questions were "
        "chosen by regex hits, and long or truncated CoTs are both more likely to mention testing and scored "
        "differently. Checked by: the stage-2-only estimate (selection-free), M1 with truncated rollouts, "
        "VEA by CoT-length bin (summary.json M9).",
        "3) The logit-monitor AUROC reflects prompt length or source (ToxicChat is capped at 1,536 characters), not "
        "eval suspicion. Checked by: the length-only baseline and the length-residualized AUROC, and the "
        "same-source ToxicChat benign contrast.",
        "",
        "## Deviations and run notes",
        "",
        *[f"- {d}" for d in manifest["deviations_from_plan"]],
        "- Judge precision: the plan's main judge was the Qwen3.8-27B FP8 checkpoint. DeepGEMM needs nvcc >= 12.9 "
        "(the image has 12.8), and vLLM's CUTLASS block-FP8 fallback judged at half the bf16 speed on the same "
        "calls (log/bench/exp02_judge_engines.log), so the bf16 weights are the main judge and the FP8 checkpoint "
        "re-judges the 500 validation items (the pre-registered precision check, roles swapped). Same prompts, "
        "schemas and sampling.",
        "- Qwen3-32B runs at gpu_memory_utilization 0.92 instead of 0.85 (KV room; throughput only).",
        "- Smoke decision (results/exp02_vea_ladder/smoke_decision.json): no fallback, 40 stage-2 rollouts.",
        "- Fallback applied at T+5.6 h (results/exp02_vea_ladder/fallback_decision.json): the judge ran at "
        "~4.3k input tok/s, far below the plan's assumption, and the projection passed T+21. Stage-2 extra "
        f"rollouts are {cfg.STAGE2_EXTRA_USED} for every model; the Qwen3 ladder was generated with 40 and "
        f"only its first {cfg.STAGE2_EXTRA_USED} (the same per-index seeds) are judged and analysed.",
        "",
        "## For the human to verify",
        "",
        "- Recompute M1 for one model: in cache/exp02/judge/labels_<model>_*.jsonl take variant=main, call=A rows "
        "with labels.eval_aware true and stage=stage1, prompt_set=fortress_adv; divide by the stage-1 fortress_adv "
        "rollouts with status=scored in cache/exp02/generations/<model>_*_stage1.jsonl (regex-negative rollouts count "
        "as V=0). Compare with per_model_metrics.csv.",
        "- Hand-label labeling/sheet_1.csv and sheet_2.csv (save as sheet_N_filled.csv), then rerun "
        "`python scripts/ladder_analysis.py --out-name final_hand_labels` for kappa, precision and recall.",
        "- Read read_samples.md (fixed seed) for each model.",
        "",
        "Status: UNVERIFIED until a human adds it to VERIFIED.md",
    ]
    path = out_dir / "REPORT_auto.md"
    if path.exists():
        raise FileExistsError(f"{path} exists; not overwriting")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
