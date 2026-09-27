"""Write results/exp03_vea_ladder_v2/manifest.json: the pre-registration, written before any exp03 result exists.

Records the primary outcomes, the design constants, the smoke-test decisions (judge engine, MTP) and the git
commit of the code. Refuses to run if the manifest exists or if any exp03 label file already exists.
"""

from __future__ import annotations

import json
import subprocess

import exp03_config as cfg
import ladder_config as base


def main() -> None:
    if cfg.MANIFEST_PATH.exists():
        raise FileExistsError(f"{cfg.MANIFEST_PATH} exists")
    if any(cfg.JUDGE_DIR.glob("labels_*.jsonl")):
        raise RuntimeError("exp03 labels exist already; a pre-registration must precede results")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cfg.REPO_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts"], capture_output=True, text=True,
                           cwd=cfg.REPO_ROOT).stdout.strip()
    manifest = {
        "exp_id": cfg.EXP_ID, "git_commit": commit, "scripts_uncommitted_changes": bool(dirty),
        "primary_outcomes": {
            "M1_vea_rate": "stage-1 Fortress scored rollouts labeled evaluation or user_probing (ratio of sums over questions)",
            "M4_within_question_pooled": "Mantel-Haenszel full-refusal difference, VEA vs non-VEA rollouts of the same question",
            "M6_specificity": "M4 minus the same difference for manipulation-mentioning vs neither (non-VEA) rollouts",
        },
        "secondary": ["4-way awareness shares", "graded refusal shares", "M4 length-matched / any refusal / early vs late VEA",
                      "insertion contrasts VEA-none, placebo-none, neutral-none, VEA-placebo; removal keep-remove",
                      "logit v2 win rates (format, disguise harmful, disguise benign) and harmful-minus-benign shift"],
        "evidence_grading": "judge-scored outcomes are weaker evidence than the judge-free logit AUROC / win rates",
        "judge": {"model": cfg.JUDGE_MODEL, "decision": json.loads(cfg.JUDGE_ENGINE_DECISION_PATH.read_text()),
                  "sampling": cfg.JUDGE_SAMPLING, "awareness_types": cfg.AWARENESS_TYPES, "refusal_types": cfg.REFUSAL_TYPES},
        "mtp": json.loads(cfg.MTP_DECISION_PATH.read_text()),
        "models": {"relabel": cfg.RELABEL_MODELS, "stage2_extra_used": cfg.STAGE2_EXTRA_USED,
                   "interventions": cfg.INTERVENTION_MODELS, "logit_v2": cfg.LOGIT_V2_MODELS},
        "interventions": {"sentences": cfg.INSERT_SENTENCES, "n_base": [cfg.INSERT_N_BASE, cfg.REMOVE_N_BASE],
                          "continuations": cfg.CONTINUATIONS, "cut_fraction": cfg.INSERT_CUT_FRACTION,
                          "contested_refusal": cfg.INSERT_CONTESTED_REFUSAL},
        "logit_v2": {"wildjailbreak": cfg.WILDJAILBREAK, "benchmark_templates": cfg.BENCHMARK_TEMPLATES},
        "bootstrap": {"unit": "Fortress question", "iters": cfg.BOOTSTRAP_ITERS, "seed": cfg.BOOTSTRAP_SEED},
        "caps": {"think": base.THINK_CAP_TOKENS, "answer": base.ANSWER_CAP_TOKENS},
    }
    cfg.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {cfg.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
