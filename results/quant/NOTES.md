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
