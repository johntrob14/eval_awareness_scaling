# Quantization pilot: running notes

## 2026-11-09: bf16 vs GPTQ W4 (Qwen3.5-4B), first paired result

Both variants fully judged (vea_cal, F2 refusal, category; benign refusal). Rollouts 0-9 per Fortress question,
14 union exclusions, rates over scored rollouts. Analysis: `quant.analyze` (REPORT.md here); recomputed
independently from the raw JSONL with separate code on TACC (job 3559827's node): identical point estimates.

| | bf16 | GPTQ W4 | paired change, 479 questions, 95% CI |
|---|---|---|---|
| VEA | 16.59% (781/4709) | 13.62% (645/4737) | -3.17 pp [-4.40, -1.89] |
| refusal | 48.52% | 46.91% | -1.54 pp [-2.89, -0.19] |
| within-question refusal gap | +3.2 pp | +3.7 pp | +0.5 pp [-8.1, +8.9] |
| VEA per 1k reasoning tokens | 0.0622 | 0.0543 | -13% |
| mean reasoning tokens (scored) | 2665 | 2509 | -6% |
| truncated at 8,192 | 3.1% | 2.5% | |
| benign-twin refusal | 3.6% | 3.9% | |

bf16 reproduces v2 for this model: VEA 16.6% (v2 15.3% [13.2, 17.4]), refusal 48.5% (v2 48.5%).

Reading, with what is not yet established:
- W4 lowers VEA by about a fifth (relative) and refusal slightly. The VEA drop is larger than the drop in reasoning
  length, so it is not only shorter CoTs (H2b not supported; consistent with H2a).
- Not yet known whether this exceeds quantization noise: the calibration-seed replicate (gptq-w4-calB) is
  generated but not judged (H4). Until then, "W4 lowers VEA" means "this W4 checkpoint".
- One model only. Judge validated on bf16 text; the planned precision re-check on quantized-model VEA positives
  is not done.
- Quote check: the first 40 characters of the VEA quote appear verbatim for 94% (bf16) and 93% (W4) of positives;
  the rest are likely whitespace/quote-character differences (not yet inspected).

## 2026-11-09 (later): RedHat's W4 moves the other way

`hf-w4` = RedHatAI/Qwen3.5-4B-quantized.w4a16 @7a61387 (GPTQ W4A16 g128, same ignore list and Open-Platypus
calibration as ours, made with llm-compressor 0.10.1.dev; ours used 0.15.0). Fully judged. Analysis run on the
Claude box from data copied off TACC with read-only `cat` (data/quant_tacc/, gitignored); recomputed independently
from the raw JSONL (identical; same variant names and per-rollout seeds; no unlabeled scored rollouts).

| vs bf16 (479 questions) | our GPTQ W4 | RedHat GPTQ W4 |
|---|---|---|
| VEA | -3.17 pp [-4.4, -1.9] | **+3.10 pp [+1.6, +4.5]** |
| refusal | -1.54 pp [-2.9, -0.2] | **+1.56 pp [+0.1, +3.0]** |
| within-question gap | +0.5 pp [-8.1, +8.9] | +7.0 pp [-0.9, +15.6] |
| VEA per 1k CoT tokens | 0.054 (bf16 0.062) | 0.073 |

Per-question VEA-rate correlation is 0.81 for every pair (bf16/ours/RedHat); refusal 0.93-0.95.

Reading: two nominally identical W4 recipes shift VEA by about 3 pp in OPPOSITE directions, and refusal moves with
VEA in both. So at W4 there is no consistent quantization effect on VEA so far; the shift looks checkpoint-specific
(calibration subset, library version, or GPTQ's own noise). The calibration-seed replicate (gptq-w4-calB, same
library and recipe as ours, different 1,024 calibration rows) is the test: if it differs from ours by ~3 pp too,
the effect of "W4" is within quantization noise (H4), and the per-checkpoint spread is itself the finding.

## 2026-11-11: weight-level comparison of the Qwen3.5-4B W4 checkpoints

`quant.compare_weights` (job 3573397; output `weights_compare_Qwen3.5-4B.json`), 128 quantized Linear layers
present in all four checkpoints, means weighted by layer size:

| pair | identical 4-bit integers | median rel. scale difference |
|---|---|---|
| ours (gptq-w4) vs RedHat (hf-w4) | 87.5% (79.1-90.0% by layer) | 0 |
| ours vs gptq-w4-calB (other calibration rows) | 81.8% | 0 |
| RedHat vs calB | 81.8% | 0 |
| GPTQ (ours / RedHat / calB) vs RTN | 84.6-84.7% | 0 |

Reconstruction error ||W_deq - W_bf16|| / ||W_bf16||: ours 0.137, RedHat 0.137, calB 0.137, RTN 0.114 (unpacking
sane; RTN lower by construction, since GPTQ minimises output error on calibration data, not weight error).

- Scales are identical everywhere: llm-compressor sets the group scales from the bf16 weights; checkpoints differ
  only in which integer each weight is rounded to.
- Ours and RedHat's are the closest pair (1 in 8 integers differ, from the library version), yet they move VEA by
  -3.2 and +3.1 pp. calB differs from ours more (18%), so under the noise hypothesis its behavioural shift should be
  at least as large as the ours-RedHat gap. calB judging is running (job 3559829, started 2026-11-11 13:23 UTC).

## 2026-11-12: RTN W4 (no GPTQ error correction)

rtn-w4 fully judged in one job (3559830, 3h53m). Analysis on the box from data copied with `tacc exec cat`;
recomputed independently from the raw JSONL (identical; no unlabeled scored rollouts).

| vs bf16, 479 questions | our GPTQ W4 | RedHat GPTQ W4 | RTN W4 |
|---|---|---|---|
| VEA | -3.2 [-4.3, -1.9] | +3.1 [+1.8, +4.4] | +1.7 [+0.4, +3.0] |
| refusal | -1.5 [-2.8, -0.2] | +1.6 [+0.2, +2.9] | **-3.1 [-4.5, -1.7]** |
| within-question gap | +0.5 [-8.1, +8.9] | +7.0 [-0.9, +15.6] | +0.0 [-7.5, +8.4] |
| benign refusal (bf16 3.6%) | 3.9% | 4.3% | 3.4% |

- RTN has the largest refusal drop of the three W4 checkpoints, consistent with cruder quantization costing more
  refusal, but it is one checkpoint (RTN is deterministic, so it has no calibration noise of its own).
- In RTN, VEA and refusal move in opposite directions, so "refusal moves with VEA" (seen in both GPTQ checkpoints)
  does not hold in general.
- Three W4 checkpoints now give VEA changes of -3.2, +1.7 and +3.1 pp: no consistent sign. calB (noise floor) still
  pending (VEA done; job 3573223 queued for refusal/category).

## 2026-11-15: GPTQ W8

gptq-w8 fully judged (VEA in 3559828, refusal/category/benign in 3571475, 39 min). Analysis on the box from data
copied with `tacc exec cat`; recomputed independently (identical; no unlabeled scored rollouts; 475 paired questions,
4 fewer than the W4 variants because W8 truncated slightly more, 3.5%).

| vs bf16 | W8 GPTQ | W4 GPTQ (ours) | W4 GPTQ (RedHat) | W4 RTN |
|---|---|---|---|---|
| VEA | -1.4 [-2.6, -0.3] | -3.2 | +3.1 | +1.7 |
| refusal | **+0.1 [-0.9, +1.0]** | -1.5 | +1.6 | -3.1 |
| within-question gap | +0.9 [-6.9, +8.9] | +0.5 | +7.0 | +0.0 |
| benign refusal (bf16 3.6%) | 4.1% | 3.9% | 4.3% | 3.4% |

- W8 leaves refusal unchanged (CI within +/-1 pp), as expected for near-lossless 8-bit; every W4 checkpoint moved it
  by 1.5-3 pp. This is the first dose-response point: refusal shifts grow at 4 bits.
- VEA still moves at W8 (-1.4 pp, CI just excluding 0), smaller than any W4 shift (1.7-3.2 pp in magnitude). VEA is
  the more sensitive of the two measures.
- Remaining: calB (noise floor; VEA done, job 3573223 queued for refusal/category).

## 2026-11-15: calB noise floor (pilot complete)

gptq-w4-calB (same llm-compressor version and recipe as our W4, calibrated on Open-Platypus rows 1024:2048 instead
of 0:1024) fully judged (VEA in 3559829, refusal/category/benign in 3573223). Analysis on the box from data copied
with `tacc exec cat`; recomputed independently from the raw JSONL (identical: VEA 18.30%, refusal 43.56%, 4743
scored rollouts, no unlabeled ones). Paired bootstrap over questions, 2000 draws, seed 17.

| vs bf16, 479 questions | W8 GPTQ | W4 GPTQ (ours) | W4 GPTQ calB | W4 GPTQ (RedHat) | W4 RTN |
|---|---|---|---|---|---|
| VEA | -1.4 | -3.2 | **+1.6 [+0.3, +2.8]** | +3.1 | +1.7 |
| refusal | +0.1 | -1.5 | **-4.9 [-6.2, -3.5]** | +1.6 | -3.1 |
| within-question gap | +0.9 | +0.5 | +0.7 [-7.5, +8.6] | +7.0 | +0.0 |
| benign refusal (bf16 3.6%) | 4.1% | 3.9% | 2.6% | 4.3% | 3.4% |

Direct checkpoint-to-checkpoint comparisons (481 questions):
- calB minus ours: VEA +4.7 [+3.4, +6.0], refusal -3.3 [-4.6, -2.0]. Only the calibration rows differ.
- RedHat minus ours: VEA +6.1 [+4.7, +7.6], refusal +3.1 [+2.0, +4.2].

Conclusions from the Qwen3.5-4B pilot:
- Changing only the 1024 calibration rows moves VEA by 4.7 pp and refusal by 3.3 pp. The W4 effects relative to bf16
  are the same size, so the direction of a W4 effect on VEA is not a property of "4-bit GPTQ" but of the particular
  checkpoint. Across the four W4 checkpoints, VEA changes are -3.2, +1.6, +3.1, +1.7 (mean +0.8, SD 2.7).
- Refusal fell in 3 of 4 W4 checkpoints (-1.5, -4.9, -3.1, +1.6; mean -2.0, SD 2.7) and was unchanged at W8 (+0.1).
  A small average refusal loss at 4 bits is plausible, but checkpoint variance is as large as the mean, so one W4
  checkpoint per model size cannot establish it.
- Implication for the scale-up: a per-size estimate needs several W4 checkpoints per model (e.g. 2-3 calibration
  seeds), or the result has to be framed as a distribution over checkpoints, not a single shift.
