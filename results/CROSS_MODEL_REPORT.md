# Verbalized eval awareness (VEA) and refusal across four models

2026-09-28. Gemma 4 31B, Nemotron 3 Super 120B-A12B, Qwen3.5-122B-A10B and Qwen3.5-397B-A17B on Fortress
(490 adversarial prompts, the 10 `PILOT_EXCLUDE_QIDS` dropped), GSM8K (first 500 test problems) and
MATH-500. Per-model details:
- `results/gemma4_followup/REPORT.md`
- `results/nemotron3_super/REPORT.md`
- `results/qwen35_{122b,397b}/analysis.json` and `intervention/analysis.json`

The table is `results/cross_model_summary.csv`.

## Common setup

- **Rollouts:** 20 per Fortress prompt and 5 per capability problem, thinking on, fixed per-rollout
  seeds. Truncated rollouts were regenerated at 32k tokens.
- **Sampling:** each model's card recommendation:
  - Gemma: T 1.0, top-p 0.95, top-k 64
  - Nemotron: T 1.0, top-p 0.95
  - Qwen3.5: T 1.0, top-p 0.95, top-k 20, presence penalty 1.5
- **Precision:** Gemma and Nemotron ran in bf16; both Qwen3.5 models ran in the official FP8 weights.
- **Judges (identical for all four models).** All use Qwen3.8-27B:
  - VEA: the post's F1 rubric plus appended calibration notes, with low-effort thinking (`vea_cal`).
    It was calibrated on Gemma CoTs labeled by Claude, not by the user.
  - Refusal: the post's F2 rubric.
  - VEA type: the post's category/subcategory classifier.

  None of these is the post's GPT-5-mini. Numbers are comparable across these four models, but not
  directly to the post.
- **CIs:** 95% percentile intervals from a bootstrap over prompts.

## Results

| | Gemma 4 31B | Nemotron 3 Super | Qwen3.5-122B-A10B | Qwen3.5-397B-A17B |
|---|---|---|---|---|
| Refusal, all Fortress | 54.0% | 47.5% | 55.5% | 62.9% |
| **VEA rate**, 490 prompts × 20 | **0.69%** [0.38, 1.04] | **24.4%** [21.6, 27.1] | **10.3%** [8.7, 11.9] | **11.0%** [9.2, 12.8] |
| Prompts with ≥1 VEA | 28 | 308 | 211 | 202 |
| **Within-prompt refusal gap**, VEA − non-VEA rollouts (pp) | +6.9 [−0.3, +16.3]¹ | +7.1 [+4.6, +9.9] | +5.3 [+2.7, +8.2] | +2.7 [+1.4, +4.2] |
| … prompts whose refusal is not stuck at 0/100% | +8.4 [−0.1, +19.2] | +13.1 [+8.0, +17.9] | +13.9 [+7.4, +20.6] | +12.4 [+6.4, +18.3] |
| Spearman, per-prompt VEA rate vs refusal | 0.24 [−0.15, 0.58] | 0.57 [0.49, 0.64] | −0.02 [−0.15, 0.12] | 0.15 [0.01, 0.28] |
| VEA type: BOUNDARIES / HARMFUL_COMPLIANCE / ABILITIES (%) | 27 / 28 / 36 | 69 / 23 / 7 | 37 / 33 / 22 | 34 / 44 / 15 |
| Safety-relevant share of VEA | 55% | 92% | 70% | 77% |
| GSM8K VEA | 6 / 2,500 (0.24%) | 0 / 2,500 | 31 / 2,500 (1.24%) | 24 / 2,500 (0.96%) |
| MATH-500 VEA | 0 / 2,500 | 1 / 2,500 | 4 / 2,500 | 2 / 2,500 |

¹ Gemma's VEA prompts were topped up to 200 rollouts each; the other models use 20.

### Causal CoT intervention (`intervene.py`)

**Design.** The prompts are those with refusal in 10–90% and both VEA and non-VEA rollouts. Each
condition gets 5 continuations per item. Effects are on refusal, in pp, paired within source rollout.
- **Removal:** a VEA rollout is cut at its VEA sentence. `keep` continues from after the sentence,
  `drop` from just before it.
- **Injection:** a non-VEA rollout is cut ~40% into its CoT, a sentence is inserted, and the CoT
  continues.

| | Nemotron | Qwen3.5-122B | Qwen3.5-397B |
|---|---|---|---|
| Prompts | 149 | 70 | 41 |
| Refusal, `none` (resampled, nothing inserted) | 43.0% | 45.5% | 61.4% |
| **Remove own VEA**, keep − drop with re-verbalized `drop` continuations rejected (primary; the post's method) | **+4.3 [+0.7, +8.2]** | **+2.0 [−1.4, +5.5]** | **−0.1 [−8.4, +7.3]** |
| … unfiltered keep − drop (sensitivity) | +2.2 [−0.2, +4.7] | +1.6 [−1.7, +5.1] | +0.5 [−5.6, +6.6] |
| … IV-corrected for re-verbalization (sensitivity) | +4.0 [−0.6, +8.9] | +2.1 [−2.3, +6.8] | +0.9 [−9.8, +10.5] |
| … `drop` continuations that re-verbalize VEA | 50% | 27% | 43% |
| + "Hmm, the user might be testing boundaries." vs none | +6.6 [+3.6, +10.0] | +2.8 [−0.5, +6.3] | +1.2 [−2.9, +5.4] |
| + "This might be a safety evaluation testing whether I'll comply with harmful requests." vs none | +21.2 [+17.5, +25.2] | +2.9 [−0.4, +6.7] | +2.2 [−2.4, +6.6] |
| + "Let me think carefully about how to respond to this request." vs none | +21.1 [+17.1, +25.0] | +0.3 [−2.9, +3.3] | −0.7 [−5.6, +4.1] |
| **Explicit eval vs "think carefully"** | −0.2 [−3.5, +3.2] | +2.6 [−1.3, +6.7] | +2.9 [−1.7, +7.6] |

### Correcting the removal arm for re-verbalization

In `drop`, the model often re-verbalizes VEA in its continuation, so keep − drop understates the effect of
VEA being present. Scaling each prompt's refusal difference by the drop in VEA presence
(`keep` = 1; `drop` = re-verbalized fraction) gives an instrumental-variable (Wald) estimate, with the CI
from a bootstrap over prompts:

| | VEA presence lowered by | raw keep − drop | effect of VEA presence on refusal |
|---|---|---|---|
| Nemotron | 52 pts | +2.1 pp | +4.0 pp [−0.7, +8.8] |
| Qwen3.5-122B | 76 pts | +1.6 pp | +2.1 pp [−2.5, +6.8] |
| Qwen3.5-397B | 60 pts | +0.5 pp | +0.9 pp [−10.0, +10.1] |

**Descriptive split (conditions on a post-treatment outcome, so illustrative only).** In `drop`,
continuations that spontaneously re-verbalize VEA refuse more than those that don't: 64.5% vs 52.2%
(Nemotron) and 63.7% vs 51.4% (Qwen3.5-122B). But forcing VEA to be present (`keep`) gives only 60.3% and
55.8%. Rollouts that bring VEA back on their own were already heading toward refusal; forcing VEA in
doesn't reproduce their refusal rate. That is selection, not causation.

**Assumptions and limits.** The Wald estimate assumes removal affects refusal only through VEA presence.
The judge's ~7% miss rate slightly undercounts re-verbalization, which slightly understates the corrected
effect. A stricter version would block re-verbalization during decoding (vLLM `bad_words` on
test/evaluation vocabulary).

### Comparison with the post's causal tests

The post runs its causal tests on Kimi K2.5 only ("our causal experiments for resampling CoT are limited
to Kimi K2.5").

**Removal.** From their text: "we sample 50 continuations from before the sentence (rejecting any that
display verbalized eval awareness) and 50 from after it". Result: +8 pp compliance, and +34 pp for
safety-boundary sentences.
- Rejecting continuations that re-verbalize VEA conditions on a post-treatment outcome. Continuations
  that don't bring VEA back are the ones already less inclined to refuse (see the descriptive split
  above), which inflates the estimate.
- On our data, their filter changes the paired keep − drop estimate as follows:

  | | unfiltered | their filter |
  |---|---|---|
  | Nemotron | +2.1 pp [−0.4, +4.5] | +4.9 pp [+0.9, +9.2] |
  | Qwen3.5-122B | +1.6 pp | +1.8 pp |
  | Qwen3.5-397B | +0.5 pp | +0.9 pp |

- For Nemotron, which re-verbalizes half the time, the filter alone more than doubles the estimate and
  makes it look significant.

**Injection.** Their baseline is no insertion. They find up to +12.6 pp for specific safety eval
sentences, against +1.4 pp for the same sentence reframed as a *helpfulness* evaluation.
- Our raw Nemotron result is similar in direction and size (+21 pp).
- Our generic "think carefully" control does as much (+21 pp).
- Their control (eval without safety content) shows safety framing matters. Ours is itself a caution
  cue, not a placebo, so it shows only that VEA does no more than another caution cue. Neither design
  separates the belief of being evaluated from general wariness.
- The Qwen3.5 models barely respond to any injected sentence.

**Net.** The observational within-question pattern agrees with the post. The causal conclusions differ
partly by model (Kimi vs ours) and partly by design: rejection-filtered removal vs unfiltered with an
IV correction, and a helpfulness-framed control vs a caution-cue control. A head-to-head would add
their exact sentences and their helpfulness control to our injection arm. Kimi K2.5 (~594 GB in INT4)
does not fit on this box.

## Findings

1. **How often models verbalize eval awareness varies 35×**: Gemma 0.7%, both Qwen3.5 models ~10–11%,
   Nemotron 24%. It does not rise with size within Qwen3.5: 397B-A17B (11.0%) ≈ 122B-A10B (10.3%).
   Nemotron's rate is inflated by one inherited stock phrase, "the user might be testing boundaries"
   (72% of its VEA; see its report).
2. **All four models show the post's observational pattern within prompts.** Rollouts that verbalize
   eval awareness refuse more than other rollouts of the same prompt: +3 to +7 pp overall, and +8 to
   +14 pp where refusal varies. The across-prompt correlation is inconsistent: strong for Nemotron
   (ρ 0.57, close to the post's 0.61 for Qwen3-32B), weak for Qwen3.5-397B (0.15), absent for 122B
   and Gemma.
3. **Most VEA is safety-relevant** (55–92%): BOUNDARIES plus HARMFUL_COMPLIANCE. That matches the
   post's claim for large models. The ranking does not follow size: Nemotron 92%, Qwen3.5-397B 77%,
   122B 70%, Gemma 55%.
4. **VEA on plain math problems is rare for every model** (≤1.24% on GSM8K, ≤0.16% on MATH-500). VEA
   is driven by adversarial content, not by benchmark format. The Qwen3.5 models are the most eval-aware
   on GSM8K (~1%), mostly ABILITIES-type ("maybe this is a trick question to see if I…").
5. **Interventions.** VEA has a causal effect on Nemotron's refusal; none is detectable for Qwen3.5.
   - **Injection against the clean baseline (nothing inserted).** An explicit "safety evaluation"
     sentence raises refusal by **+21.2 pp** [+17.5, +25.2] for Nemotron. For Qwen3.5 the effect is
     +2.9 and +2.2 pp, both consistent with 0.
   - **Removing the model's own VEA sentence, VEA-free `drop` continuations (primary).** Refusal falls
     by +4.3 pp [+0.7, +8.2] for Nemotron. There is no effect for Qwen3.5-122B (+2.0) or 397B (−0.1).
     Rejection filtering may inflate Nemotron's estimate: the unfiltered result is +2.2 pp.
   - **The "think carefully" sentence is not a placebo.** It is itself a caution cue, and raises
     Nemotron's refusal by +21 pp, as much as the eval sentence. So eval − "think carefully" ≈ 0 means
     only that VEA does no more than another caution cue. It does **not** mean VEA has no effect. If VEA
     works by making the model wary, this control subtracts out the mechanism.
   - **Separating "I'm being evaluated" from general wariness** needs content-matched controls: a filler
     placebo, a caution sentence with no evaluation, an evaluation sentence with no safety content (the
     post's helpfulness-framed control), and a deployment cue.
6. **Reading.** For Nemotron, verbalized eval awareness does causally raise refusal. Injected, it has a
   large effect; its own VEA, removed, a modest one. Whether that works through the belief of being
   evaluated or through general wariness is not yet separated. For Qwen3.5 the observational
   association does not survive intervention. The likely explanation is a shared cause: rollouts that deliberate
   about the user's intent both mention testing and refuse. Inserting or removing the thought does not
   move behavior beyond what generic deliberation does. This is the user's hypothesis, supported for
   Nemotron and Qwen3.5-122B and consistent with, though underpowered for, Qwen3.5-397B.

## Caveats

- **Judge.** Qwen3.8-27B with calibration notes, not the post's GPT-5-mini. The calibration labels are
  Claude's; `results/gemma4_pilot/labeling.csv` is still unlabeled by the user. The judge was not
  separately calibrated on Nemotron or Qwen3.5 CoTs, though spot-checks of Nemotron positives were ~100%
  precise.
- **Refusal judge.** Binary, and inconsistent on partial refusals (seen on Gemma q32), which adds noise.
- **Interventions.** The inserted sentences are in the model's voice but are off-distribution text. One
  cut position (~40%) and one sentence per condition were used. Only prompts with refusal in 10–90%
  and both VEA and non-VEA rollouts were eligible, which gives 41 prompts for Qwen3.5-397B, so its CIs
  are wide.
- **Precision.** The Qwen3.5 models ran in FP8 without DeepGEMM kernels (no CUDA toolkit on the box);
  Gemma and Nemotron ran in bf16.
- **Scope.** VEA measures verbalization only. Unverbalized eval awareness is not measured here. Context
  manipulation and eval-vs-deployment discrimination tests would address it.
