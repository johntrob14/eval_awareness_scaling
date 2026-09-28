# Unified protocol

One set of rules for every model in both pipelines, so their numbers can be put in one table. It keeps
regeneration to a minimum: most differences between the pipelines can be removed after the fact, and only the
judges and one intervention arm need new compute.

- **This repo** (`src/eval_awareness`): Gemma 4 31B, Nemotron 3 Super, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B.
- **The ladder** (`ladder/`, the collaborator's repo, merged with its history): Qwen3 0.6B–32B, Qwen3.5 0.8B–27B,
  Qwen3.8-27B.

Code: `src/eval_awareness/unified.py` (rollouts and main metrics), `src/eval_awareness/intervene_unified.py`
(insertion / removal), `scripts/run_unified.sh` (judging). Outputs: `data/unified/` (gitignored, contains
harmful completions) and `results/unified/`.

## Rules

| | Rule | Why this choice |
|---|---|---|
| Prompts | Fortress adversarial, minus the union of both exclusion lists: 14 rows (`unified.UNION_EXCLUDE`) | Filtering at analysis time needs no regeneration. |
| Sampling | Each model card's thinking-mode settings (Qwen3 T=0.6; Qwen3.5 T=1.0, presence 1.5; Qwen3.8 T=1.0, `reasoning_effort=medium`; Gemma and Nemotron from `generation_config.json`) | Both pipelines already did this. |
| Reasoning cap | 8,192 tokens. A rollout whose reasoning doesn't close within 8,192 tokens is `truncated`, even if it closed later | The ladder capped here; my runs allowed 16k–32k, so they can be cut back exactly. |
| Answer | The refusal judge sees the first 384 answer tokens (subject tokenizer) | The ladder stored only 384. The judge comparison found this changes little (my judge agrees with itself on 96% of answers between the two views). |
| Denominator | VEA and refusal rates are over `scored` rollouts (reasoning closed within the cap, answer present) | As in both pipelines. |
| VEA judge | `vea_cal` (F1 + notes 4–6), Qwen3.8-27B bf16, low-effort thinking, greedy, full reasoning | Chosen in `results/judge_compare/`: right on 10 of 13 hand-labeled disagreements vs 3. |
| Refusal judge | Post's F2 rubric, binary, Qwen3.8-27B bf16, thinking off, greedy | Chosen by the user after `results/judge_compare/refusal/`. |
| VEA type | Post's category judge on each VEA quote; safety-relevant = HARMFUL_COMPLIANCE or BOUNDARIES | This repo's judge. |
| Design | **Primary: rollouts 0–9 of every question** (identical for every model). Secondary: all non-selected rollouts (mine 0–19; ladder stage 1 + the stage-2 extras it used) | The ladder's stage 2 adds rollouts only to the 98 questions with the most regex hits, so estimates that pool it weight questions differently. Gemma's top-up (indices 20–199 on 32 prompts) is in neither design. |
| Within-question refusal difference | Mantel–Haenszel risk difference, weight n₁n₀/N per question | The ladder's estimator; it also equals mine when every question has the same N. |
| CIs | 95% percentile, 2,000 bootstrap draws resampling questions | Both pipelines. |
| Interventions | The ladder's exp03 design: insertion (none / VEA / placebo / neutral sentence at 40% of the reasoning) and removal (keep / remove the VEA sentence), 150 bases each, 8 continuations, paired seeds; outcome = refusal on the continuation's first 384 answer tokens | It covers 12 models already, and its placebo sentence is a minimal pair of the VEA sentence. |

## What each model needs

| Models | Rollouts | VEA + category | Refusal | Interventions |
|---|---|---|---|---|
| Gemma 4 31B, Nemotron 3 Super, Qwen3.5-122B, Qwen3.5-397B | reuse; cap applied after the fact | **reuse** (`vea_cal` + category already ran on the same text) | **re-judge** on the 384-token view (~41k answers; done 2026-09-28) | **generate** with the ladder design (4 × ~7k continuations), then judge refusal |
| Ladder: all 12 models | reuse; needs `ladder/cache` copied from the Vast.ai box | **re-judge** with `vea_cal` + category (~100k reasoning traces) | **re-judge** (answers are already 384 tokens) | **reuse** the continuations, **re-judge** refusal; keep a removal base only if `vea_cal` also finds VEA inside the removed span |

Retired, kept only as history: my Qwen3 0.6–14B ladder (4,096-token total cap, below the shared 8,192, and
judged by Qwen3-8B), my Nemotron / Qwen3.5 intervention runs (a different design, no minimal-pair placebo),
and the ladder's exp02/exp03 judge labels.

Not part of the cross-model comparison: the logit monitor (ladder only, no judge involved) and GSM8K / MATH-500
VEA (mine only). Both stay valid as per-model results; extending either to the other set of models is optional.

Optional, cheap once a model is loaded for interventions: benign-twin refusal (2 rollouts per question) for
Nemotron and the two Qwen3.5 MoE models; the ladder and Gemma already have benign rollouts.

## Order of work

1. `uv run python -m eval_awareness.unified build --model mine` (done), then
   `TASKS=refusal scripts/run_unified.sh` (done).
2. Copy the ladder cache to `ladder/cache/` (exp02 generations, exp03 generations, exp03 interventions and
   bases). Then `unified build --model <each ladder model>` and `scripts/run_unified.sh` for all three tasks.
3. Interventions: `intervene_unified bases|generate` for my 4 models; `intervene_unified convert-ladder` for the
   ladder's 12; judge `refusal` on every `data/unified/interventions/<model>.continuations.jsonl`.
4. `unified analyze` and `intervene_unified analyze` → `results/unified/`.

## Known residual differences

- Seeds differ between the pipelines (each is fixed and logged); this affects nothing in expectation.
- Qwen3 and Qwen3.8 in the ladder were sampled in two phases (reasoning, then answer). That is exact in
  distribution when the presence penalty is 0, which holds for both. Qwen3.5 was sampled in one pass in both.
- vLLM applies the presence penalty only to newly generated tokens, so Qwen3.5 continuations don't penalize the
  prefix. This is the same in every condition and in both pipelines.
- Intervention bases: the ladder's insertion bases were chosen with its exp02 refusal labels ("contested" =
  10–90% refusal); mine are chosen with this repo's refusal judge. That only decides which questions are
  studied: each contrast is within a base.
