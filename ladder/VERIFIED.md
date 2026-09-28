# Verified results

A result is established only once it is listed here by a human (see `AGENTS.md`). Each entry names the commit,
the check, and who did it. Verification materials: `results/exp03_vea_ladder_v2/verification/`.

## exp03_vea_ladder_v2 @ 832896e

- **Insertion arithmetic.** Qwen3-32B contrasts recomputed from per-rollout judge labels
  (`verification/recompute_qwen3_32b_insertion.csv`): VEA-none +25.6, placebo-none +27.0, VEA-placebo -1.4 points,
  matching `final/REPORT_auto.md`. Z.P., 9/28/26
- **Judge refusal labels.** Human vs judge on 39 insertion answers (Qwen3-32B, Qwen3-14B; stratified 13 per judge
  label; `verification/agreement.md`): full-refusal binary kappa 0.63 [0.32, 0.87], 3-way kappa 0.54.
  All 12 disagreements involve "partial"; none is full refusal vs comply. The judge calls full refusal where the
  human saw partial in 5 of 13 cases, so absolute full-refusal rates are likely inflated. Z.P., 9/28/26
- **VEA ≈ placebo on refusal (Qwen3-32B, Qwen3-14B).** Holds with full refusal as the outcome (-1.4 [-6.1, +3.5];
  -2.4 [-6.4, +1.4]) and with any refusal, which is unaffected by the judge's full-vs-partial confusions
  (-0.2 [-5.4, +4.6]; -1.4 [-5.4, +2.8]). Judge-scored. Z.P., 9/28/26

## Not yet verified

- exp03 judge awareness (VEA) labels: `verification/awareness_labels.csv` not yet filled (checklist task 2).
  M1, M4 and M6 depend on these labels.
- exp01 and exp02 results.
- Splice read: (`verification/splice_examples.md`, 4 Qwen3-32B questions): <one line on what you saw>.
  <initials>, <date>