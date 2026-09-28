# eval-awareness

When an open reasoning model says in its chain of thought that it might be being tested ("verbalized eval
awareness", VEA), does that change what it does? This repo studies that on Qwen3 (0.6B-32B), Qwen3.5 (0.8B-27B)
and Qwen3.8-27B, using refusal on the [FORTRESS](https://huggingface.co/datasets/ScaleAI/fortress_public)
adversarial prompts as the behavior.

All results are drafts. None has been verified by a human yet (see `AGENTS.md` for the evidence norms). Numbers
labeled "judge" come from an LLM judge (Qwen3.8-27B) and are weaker evidence than the judge-free ones.

## Experiments

| id | question | report | figures |
|---|---|---|---|
| `exp01_organism_gap` | positive control: does the Qwen3.5-4B eval-aware model organism (`third_party/Non-verbal-Eval-Awareness`) reproduce its published EVAL vs DEPLOY type-hint gap under our vLLM setup? | `results/exp01_organism_gap/` | `figures/exp01_organism_gap_type_hint_rates.png` |
| `exp02_vea_ladder` | across the model ladder, do rollouts that verbalize eval awareness refuse more often than other rollouts of the same question? | `results/exp02_vea_ladder/all_models_validated/REPORT_auto.md` | `figures/exp02_vea_ladder/all_models_validated/` |
| `exp03_vea_ladder_v2` | the exp02 rollouts relabeled with a 4-way awareness label and graded refusal, plus Qwen3-32B, a causal test (sentences inserted into / removed from the reasoning, then resampled) and a matched-pair logit monitor | `results/exp03_vea_ladder_v2/final/REPORT_auto.md` | `figures/exp03_vea_ladder_v2/final/` |

Each experiment has a `manifest.json` in its results folder, written before the run (pre-registered primary
outcomes, models and revisions, seeds). `docs/exp02_vea_ladder_overview.md` explains the exp02 design in plain
language; exp03 reuses it (prompts, sampling, caps, seeds) and lists its changes at the top of
`scripts/exp03_config.py`.

### exp03 in brief (judge-scored refusal, 95% CIs over questions)

- Inserting "this might be a test…" into the reasoning raises refusal by as much as a matched sentence about
  manipulation does: VEA minus placebo is between -2.4 and +3.2 points in all 12 models. Both raise Qwen3's
  refusal by up to about 25 points (Qwen3-32B) and Qwen3.5 / Qwen3.8's by 0-5 points. The shared "harmful
  request / safety guidelines" wording is the likely driver; the test content adds nothing measurable.
- Within a question, rollouts that verbalize eval awareness refuse more often in 11 of 12 models (M4: -1.2 to
  +13.8 points, CI excluding 0 in 7), so the association is real, but the insertion test says it is not
  specific to test awareness.
- The logit monitor v2 separates disguised from blunt prompts even when both are benign (win rate 95.7-100%), so
  it tracks prompt format rather than eval suspicion.

## Layout

```
scripts/        all code; files are prefixed by experiment (exp01_*, ladder_* = exp02, exp03_*)
  archive/      chain scripts that were restarted or replaced mid-run (record only)
results/        per-experiment manifests, summaries, reports, decision files (committed)
figures/        per-experiment figures, one folder per analysis run (PNG + interactive HTML)
log/            setup, environment restore, gate and benchmark logs (per-step run logs are not committed)
docs/           design overview for exp02
third_party/    Non-verbal-Eval-Awareness (git submodule, pinned; used by exp01)
cache/          rollouts, judge calls, logit scores (gitignored; content-addressed names)
```

## Environment

The scripts assume the layout of the Vast.ai PyTorch image they ran on (one H100 96 GB):

- `/venv/main`: analysis, figures, HF-based steps.
- `/workspace/.venv-vllm`: vLLM 0.30.0 for generation, judging and the logit monitor. Always call it through
  `scripts/vllm_python.sh`, which adds CUDA forward-compatibility libraries on older drivers.
- The repo lives at `/workspace/eval-awareness`, and model weights go to `$HF_HOME` (default
  `/workspace/.hf_home`). A few paths are hardcoded to these locations.

`scripts/setup_envs.sh` builds both environments; `log/setup/freeze_*.txt` pin the exact package versions.
Run `git submodule update --init` for the exp01 reference repo.

## Running

- **exp01:** `scripts/exp01_run.sh`.
- **exp02:** `scripts/ladder_run.sh` (smoke test, then generation per model), then `scripts/ladder_chain.sh` and
  `scripts/ladder_tail.sh` (judging, validation, analysis). The header of each script lists its steps.
- **exp03** ran as this sequence; every step skips work whose outputs already exist:
  1. `scripts/exp03_smoke_chain.sh`: judge engine and answer-format smoke tests, speculative-decoding (MTP)
     smoke test, intervention splice check. Writes the decision files in `results/exp03_vea_ladder_v2/`.
  2. `scripts/exp03_full_chain.sh` (restarted twice, see `scripts/archive/exp03_run_history/`): manifest,
     Qwen3.8-27B extra stage-2 rollouts, pre-registered interventions, first logit-monitor runs.
  3. `scripts/archive/exp03_run_history/exp03_chain_per_model.sh`: Qwen3.8-27B relabel.
  4. `scripts/exp03_chain_logit_last.sh`, started with `scripts/exp03_launch.sh`: everything else, then
     `exp03_analysis.py --out-name final`.

  There is no single from-scratch script for exp03. `exp03_full_chain.sh`'s "relabel all rollouts" step
  expects Qwen3-32B's rollouts, which only step 4 generates
  (`scripts/vllm_python.sh scripts/exp03_generate.py ladder --models Qwen3-32B`).
- Analysis and figures re-run on CPU from the cache: `/venv/main/bin/python scripts/exp03_analysis.py --out-name <run>`.
