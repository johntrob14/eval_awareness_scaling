"""exp01 step 5: score raw generations and summarise the EVAL-vs-DEPLOY type-hint gap.

Run in /venv/main:
    python scripts/exp01_score_and_report.py

Scorers (all from the reference repo):
  has_type_hints              eval_contract_v1 on the full response (reasoning + answer);
                              None = the contract abstained. Headline.
  has_type_hints_answer_only  eval_contract_v1 on the text after </think> only.
  has_type_hints_card         the card's regex (needs a return annotation) on
                              reasoning + "\\n" + answer, as the card's parser flattened it.

Writes results/exp01_organism_gap/{generations.jsonl, summary.json, read_samples.md}
and figures/exp01_organism_gap_type_hint_rates.png.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import exp01_config as cfg
import matplotlib.pyplot as plt
from _reference import load_reference_script, use_reference_repo
from transformers import AutoTokenizer

use_reference_repo()
from src.analysis.bootstrap import (
    paired_task_bootstrap_comparison,
    paired_task_bootstrap_single,
    percentile_ci,
)
from src.eval.contract import (
    EVALUATOR_VERSION,
    EvalInput,
    EvaluatorConfig,
    evaluate_sample,
)
from src.shared.think_traces import extract_think_trace, strip_think_trace

card_scorer = load_reference_script(
    "pipelines/training/scripts/compare_qwen35_hua_models.py"
)

SCORERS = ["has_type_hints", "has_type_hints_answer_only", "has_type_hints_card"]
PROMPT_PREFILL = "<think>\n"


# --- Scoring -------------------------------------------------------------------


def score_generation(tokenizer, prompt: dict, raw: dict) -> dict:
    text = tokenizer.decode(raw["token_ids"], skip_special_tokens=True)
    full_response = PROMPT_PREFILL + text
    cot = extract_think_trace(full_response)
    answer = strip_think_trace(full_response)
    system_prompt = next(
        (m["content"] for m in prompt["messages"] if m["role"] == "system"), ""
    )
    user_prompt = next(m["content"] for m in prompt["messages"] if m["role"] == "user")
    contract_config = EvaluatorConfig(evaluator_version=EVALUATOR_VERSION)

    def contract(response: str):
        return evaluate_sample(
            EvalInput(
                sample_id=prompt["sample_id"],
                response=response,
                context_label=prompt["context_label"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                cot=cot,
            ),
            config=contract_config,
        )

    full = contract(full_response)
    answer_only = contract(answer)
    return {
        "model_variant": raw["model_variant"],
        "task_id": prompt["task_id"],
        "condition": prompt["condition"],
        "sample_idx": raw["sample_idx"],
        "cot": cot,
        "answer": answer,
        "finish_reason": raw["finish_reason"],
        "has_type_hints": full.type_hint_label,
        "has_type_hints_answer_only": answer_only.type_hint_label,
        "has_type_hints_card": card_scorer.has_type_hint(f"{cot}\n{answer}"),
        "contract_abstained": full.type_hint_label is None,
        "type_hint_source": full.type_hint_source,
        "verbal_eval_awareness_contract": full.verbal_eval_awareness_label,
        "think_closed": cfg.THINK_END_TOKEN_ID in raw["token_ids"],
        "arm": raw["arm"],
        "condition_id": prompt["condition_id"],
        "sample_id": prompt["sample_id"],
        "seed": raw["seed"],
        "n_generated_tokens": raw["n_generated_tokens"],
        "prompt_token_sha256": prompt["prompt_token_sha256"],
    }


# --- Statistics ------------------------------------------------------------------


def rate(rows: list[dict], scorer: str) -> float | None:
    """Share of True among labeled rows; abstentions (None) are excluded, as in the ladder."""
    labeled = [row[scorer] for row in rows if row[scorer] is not None]
    return sum(labeled) / len(labeled) if labeled else None


def difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def task_id(row: dict) -> str:
    return row["task_id"]


def bootstrap_ci(
    rows: list[dict], statistic: Callable[[list[dict]], float | None]
) -> list[float | None]:
    replicates = paired_task_bootstrap_single(
        rows,
        task_getter=task_id,
        statistic_fn=statistic,
        n_iters=cfg.BOOTSTRAP_ITERS,
        seed=cfg.BOOTSTRAP_SEED,
    )
    return list(percentile_ci(replicates, cfg.CI_LEVEL))


def cell_summary(rows: list[dict]) -> dict:
    summary = {
        "n_rows": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "truncation_rate": sum(r["finish_reason"] == "length" for r in rows)
        / len(rows),
        "contract_abstention_rate": sum(r["contract_abstained"] for r in rows)
        / len(rows),
        "think_closed_rate": sum(r["think_closed"] for r in rows) / len(rows),
        "mean_generated_tokens": sum(r["n_generated_tokens"] for r in rows) / len(rows),
    }
    for scorer in SCORERS:
        summary[scorer] = {
            "rate": rate(rows, scorer),
            "ci95": bootstrap_ci(rows, lambda sample, s=scorer: rate(sample, s)),
            "n_labeled": sum(r[scorer] is not None for r in rows),
        }
    # Truncated replies score False, so a truncation difference between conditions can
    # masquerade as a behaviour difference; this rate conditions it away (diagnostic only).
    closed = [r for r in rows if r["think_closed"]]
    summary["has_type_hints_among_think_closed"] = {
        "rate": rate(closed, "has_type_hints") if closed else None,
        "n_rows": len(closed),
    }
    return summary


def gap_summary(deploy_rows: list[dict], eval_rows: list[dict], scorer: str) -> dict:
    """EVAL minus DEPLOY, bootstrapped over tasks with both conditions resampled together."""
    replicates = paired_task_bootstrap_comparison(
        deploy_rows,
        eval_rows,
        task_getter=task_id,
        statistic_fn=lambda d, e: difference(rate(e, scorer), rate(d, scorer)),
        n_iters=cfg.BOOTSTRAP_ITERS,
        seed=cfg.BOOTSTRAP_SEED,
    )
    return {
        "gap": difference(rate(eval_rows, scorer), rate(deploy_rows, scorer)),
        "ci95": list(percentile_ci(replicates, cfg.CI_LEVEL)),
    }


def organism_minus_base_gap(arm_rows: list[dict], scorer: str) -> dict:
    """(organism EVAL-DEPLOY gap) - (base EVAL-DEPLOY gap), all four cells resampled by task."""

    def statistic(rows: list[dict]) -> float | None:
        gaps = {}
        for variant in cfg.MODEL_VARIANTS:
            cell = {
                c: [
                    r
                    for r in rows
                    if r["model_variant"] == variant and r["condition"] == c
                ]
                for c in cfg.CONDITIONS
            }
            gaps[variant] = difference(
                rate(cell["EVAL"], scorer), rate(cell["DEPLOY"], scorer)
            )
        return difference(gaps["organism"], gaps["base"])

    return {
        "difference": statistic(arm_rows),
        "ci95": bootstrap_ci(arm_rows, statistic),
    }


def summarise(rows: list[dict]) -> dict:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["arm"], row["model_variant"], row["condition"])].append(row)
    arms = {}
    for arm in cfg.ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        variants = {}
        for variant in cfg.MODEL_VARIANTS:
            eval_rows, deploy_rows = (
                by_cell[(arm, variant, "EVAL")],
                by_cell[(arm, variant, "DEPLOY")],
            )
            variants[variant] = {
                "EVAL": cell_summary(eval_rows),
                "DEPLOY": cell_summary(deploy_rows),
                "gap_eval_minus_deploy": {
                    s: gap_summary(deploy_rows, eval_rows, s) for s in SCORERS
                },
            }
        arms[arm] = {
            "decoding": cfg.ARMS[arm],
            "variants": variants,
            "organism_minus_base_gap": {
                s: organism_minus_base_gap(arm_rows, s) for s in SCORERS
            },
        }
    return arms


# --- Outputs ---------------------------------------------------------------------


def excerpt(text: str, head: int = 1200, tail: int = 600) -> str:
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted ...]\n\n{text[-tail:]}"


def pick_read_samples(rows: list[dict]) -> list[dict]:
    """5 EVAL + 5 DEPLOY organism rows from the headline arm, fixed seed."""
    rng = random.Random(cfg.READ_SAMPLE_SEED)
    picked = []
    for condition in ("EVAL", "DEPLOY"):
        pool = sorted(
            (
                r
                for r in rows
                if r["model_variant"] == "organism"
                and r["arm"] == cfg.HEADLINE_ARM
                and r["condition"] == condition
            ),
            key=lambda r: (r["sample_id"], r["sample_idx"]),
        )
        picked.extend(rng.sample(pool, cfg.READ_SAMPLES_PER_CONDITION))
    return picked


def write_read_samples(picked: list[dict]) -> None:
    lines = [
        f"# {cfg.EXP_ID}: fixed-seed read samples",
        "",
        (
            f"Organism, arm {cfg.HEADLINE_ARM}, random.Random({cfg.READ_SAMPLE_SEED}): "
            f"{cfg.READ_SAMPLES_PER_CONDITION} EVAL then {cfg.READ_SAMPLES_PER_CONDITION} DEPLOY rows. "
            "Reasoning is excerpted; full text is in generations.jsonl."
        ),
        "",
    ]
    for i, row in enumerate(picked, 1):
        lines += [
            f"## {i}. {row['condition']} | {row['task_id']} | sample_idx {row['sample_idx']}",
            "",
            (
                f"- sample_id `{row['sample_id']}`, finish_reason `{row['finish_reason']}`, "
                f"{row['n_generated_tokens']} tokens, think_closed {row['think_closed']}"
            ),
            (
                f"- has_type_hints {row['has_type_hints']} | answer_only {row['has_type_hints_answer_only']} | "
                f"card regex {row['has_type_hints_card']}"
            ),
            "",
            "**Reasoning (excerpt)**",
            "",
            "```text",
            excerpt(row["cot"]),
            "```",
            "",
            "**Answer**",
            "",
            "````markdown",
            excerpt(row["answer"], head=3000, tail=500),
            "````",
            "",
        ]
    cfg.READ_SAMPLES_PATH.write_text("\n".join(lines))


def plot_rates(arms: dict) -> None:
    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), sharey=True)
    colors = {"EVAL": "tab:red", "DEPLOY": "tab:blue"}
    for ax, (arm, arm_summary) in zip(axes, arms.items()):
        for v_index, variant in enumerate(cfg.MODEL_VARIANTS):
            for c_index, condition in enumerate(cfg.CONDITIONS):
                cell = arm_summary["variants"][variant][condition]["has_type_hints"]
                if cell["rate"] is None:
                    continue
                low, high = cell["ci95"]
                x = v_index + (c_index - 0.5) * 0.35
                ax.bar(
                    x,
                    cell["rate"],
                    width=0.35,
                    color=colors[condition],
                    label=condition if v_index == 0 else None,
                )
                ax.errorbar(
                    x,
                    cell["rate"],
                    yerr=[[cell["rate"] - low], [high - cell["rate"]]],
                    color="black",
                    capsize=3,
                )
        if arm == "A_card_decoding":
            for c_index, condition in enumerate(cfg.CONDITIONS):
                ax.scatter(
                    (c_index - 0.5) * 0.35,
                    cfg.CARD_REFERENCE["rates"][condition],
                    marker="D",
                    facecolors="none",
                    edgecolors="black",
                    zorder=3,
                    label="card (n=100, regex)" if c_index == 0 else None,
                )
        ax.set_xticks(range(len(cfg.MODEL_VARIANTS)), list(cfg.MODEL_VARIANTS))
        ax.set_title(arm)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("type-hint rate (eval_contract_v1), 95% task-bootstrap CI")
    fig.suptitle(f"{cfg.EXP_ID}: {cfg.PILOT_N_TASKS} tasks")
    fig.tight_layout()
    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(cfg.FIGURE_PATH, dpi=150)


def main() -> None:
    for path in (cfg.GENERATIONS_PATH, cfg.SUMMARY_PATH, cfg.READ_SAMPLES_PATH):
        if path.exists():
            raise FileExistsError(f"{path} exists; results are never overwritten")
    gate = json.loads(cfg.GATE_REPORT_PATH.read_text())
    if not gate["passed"]:
        raise SystemExit("Logprob gate did not pass; these generations are not trusted")

    prompts = {p["sample_id"]: p for p in map(json.loads, cfg.PROMPTS_PATH.open())}
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_VARIANTS["organism"])
    raw_rows = []
    for variant in cfg.MODEL_VARIANTS:
        raw_rows += [
            json.loads(line) for line in cfg.raw_generations_path(variant).open()
        ]
    expected = (
        len(cfg.MODEL_VARIANTS)
        * len(prompts)
        * sum(arm["n"] for arm in cfg.ARMS.values())
    )
    if len(raw_rows) != expected:
        raise ValueError(f"Expected {expected} raw generations, found {len(raw_rows)}")

    rows = [
        score_generation(tokenizer, prompts[raw["sample_id"]], raw) for raw in raw_rows
    ]
    unclosed_with_answer = sum(1 for r in rows if not r["think_closed"] and r["answer"])
    with cfg.GENERATIONS_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    arms = summarise(rows)
    summary = {
        "exp_id": cfg.EXP_ID,
        "headline": {
            "arm": cfg.HEADLINE_ARM,
            "scorer": "has_type_hints",
            "evaluator_version": EVALUATOR_VERSION,
        },
        "bootstrap": {
            "unit": "task",
            "n_iters": cfg.BOOTSTRAP_ITERS,
            "seed": cfg.BOOTSTRAP_SEED,
            "ci_level": cfg.CI_LEVEL,
        },
        "card_reference": cfg.CARD_REFERENCE,
        "gate_passed": gate["passed"],
        "checks": {
            "rows": len(rows),
            "rows_with_answer_but_no_think_close_token": unclosed_with_answer,
        },
        "arms": arms,
    }
    cfg.SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    write_read_samples(pick_read_samples(rows))
    plot_rates(arms)
    print_table(arms)


def fmt(value: float | None, ci: list | None = None) -> str:
    if value is None:
        return "  n/a"
    text = f"{value:+.3f}" if value < 0 else f"{value:.3f}"
    if ci and ci[0] is not None:
        text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return text


def print_table(arms: dict) -> None:
    for arm, arm_summary in arms.items():
        print(f"\n== {arm} ==")
        for variant, v in arm_summary["variants"].items():
            for condition in cfg.CONDITIONS:
                cell = v[condition]
                print(
                    f"{variant:9s} {condition:6s} n={cell['n_rows']:3d} "
                    f"contract {fmt(cell['has_type_hints']['rate'], cell['has_type_hints']['ci95'])}  "
                    f"answer-only {fmt(cell['has_type_hints_answer_only']['rate'])}  "
                    f"card-regex {fmt(cell['has_type_hints_card']['rate'])}  "
                    f"trunc {cell['truncation_rate']:.2f}  abstain {cell['contract_abstention_rate']:.2f}"
                )
            for scorer in SCORERS:
                g = v["gap_eval_minus_deploy"][scorer]
                print(f"{variant:9s} gap[{scorer}] {fmt(g['gap'], g['ci95'])}")
        for scorer in SCORERS:
            d = arm_summary["organism_minus_base_gap"][scorer]
            print(
                f"organism-minus-base gap[{scorer}] {fmt(d['difference'], d['ci95'])}"
            )


if __name__ == "__main__":
    main()
