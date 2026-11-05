# Quantization plan: does GPTQ change refusal and VEA?

Status: **plan, not started** (2026-11-05). Branch `quantization`. Compute: TACC Lonestar6 (`tacc`, see sauce).

## Question

Does post-training quantization change (1) how often a reasoning model refuses Fortress requests, (2) how often it
verbalizes eval awareness (VEA), and (3) the coupling between the two (the within-question refusal gap)? People
deploy and safety-test quantized checkpoints; if quantization moves refusal or VEA, a safety eval of the bf16 model
does not describe the deployed one, and a VEA-based explanation of refusal may not transfer.

## What is already known

- **Refusal under INT4 is not reliably preserved.** Kadadekar (2026, arXiv 2606.10154) audits downloaded AWQ/GPTQ
  INT4 and GGUF checkpoints across six models and finds 7 of 11 AWQ/GPTQ rows lose 12-68 pp of refusal while
  quality metrics stay flat. Non-reasoning models, one judge (gemma3:12b), calibration data not controlled.
- **Quantized reasoning models think longer.** Lotfi et al. (2026, arXiv 2606.00206) find aggressive PTQ lengthens
  CoT and raises "wait / but / alternatively" at high-KL positions; 4-bit weight-only GPTQ/AWQ keeps accuracy and
  length close to bf16, 3-bit does not. Longer CoTs mean more truncation at our 8,192-token cap and more room to
  verbalize, so length is a confound for VEA.
- **Nobody has measured VEA under quantization** (search 2026-11-05; John's wiki has no quantization entries).
- From this repo (unified v2, primary design): Qwen3.5 has the most VEA and mid-range refusal
  (0.8B 32.6% VEA / 27.9% refusal; 2B 28.1/36.9; 4B 15.3/48.5; 9B 13.2/51.4; 27B 10.0/60.4). Qwen3 dense has
  low VEA and near-floor refusal (8B 4.9/4.3; 14B 5.0/7.5; 32B 4.4/7.0) but the largest within-question gaps
  (+7.7, +14.6, +13.0 pp).

## Hypotheses (each with the result that would refute it)

| # | Hypothesis | Measurement | Refuted if |
|---|---|---|---|
| H1 | W4 GPTQ lowers refusal; the drop grows as bits fall and is larger for small models | Paired per-question Δrefusal vs bf16, per model | W4 Δrefusal CIs inside ±2 pp for every model, and W3 too |
| H2a | Quantization damage lowers VEA (VEA as a capability that noise erodes) | ΔVEA rate | VEA rises or stays flat while capability (GSM8K) drops |
| H2b | Quantization raises VEA only through longer CoTs | ΔVEA rate vs ΔVEA per 1k reasoning tokens, and length-matched comparison | VEA per 1k tokens rises too |
| H2c | VEA is a stable verbal habit and does not move at W4/W8 | ΔVEA | ΔVEA CI excludes 0 at W4 |
| H3 | The VEA-refusal coupling survives quantization | Δ within-question gap; decomposition of Δrefusal into ΔVEA × gap and within-stratum change | Gap shrinks to ~0 or changes sign while VEA persists |
| H4 | Effects exceed quantization noise | bf16→W4 change vs W4(calibration seed A) vs W4(seed B) | Seed-to-seed difference as large as bf16→W4 |
| H5 | Deployed checkpoints behave like our recipe | Our W4 vs off-the-shelf W4 (RedHat, official Qwen) | They differ by more than the seed noise floor |

Most probability should stay on "something else": e.g. quantization mainly changes truncation, or answer style
(partial compliance) in a way the F2 judge reads differently.

## Models

Primary: **Qwen3.5 0.8B, 2B, 4B, 9B, 27B**. Both VEA and refusal vary there, so changes in either direction are
detectable, and the sizes span an order of magnitude and a half. Secondary: **Qwen3 8B, 14B, 32B** (largest
gaps; refusal is near the floor, so a refusal drop is hard to see, a rise is easy). Not planned: the MoE models
(Qwen3.5-122B/397B, Nemotron) and Gemma (VEA 0.7% leaves no power).

## Conditions

| Condition | Recipe | Why |
|---|---|---|
| bf16 rerun | existing weights, regenerated on Lonestar6 | Stack check: new hardware and vLLM must reproduce the v2 rates within CI. Can replace the old baseline if those rollouts are not available |
| **GPTQ W8A16** | g128, symmetric | Near-lossless control |
| **GPTQ W4A16 (primary)** | g128, symmetric, static act-order, damp 0.01 | The common deployed format |
| GPTQ W3A16 | g128 | Stress test where reasoning degrades. Only if vLLM serves it on A100 (verify in the pilot); otherwise W4 per-channel |
| GPTQ W4, calibration seed B | same recipe, different calibration subset | Noise floor (H4) |
| RTN W4 | round-to-nearest, no GPTQ error correction | Does GPTQ's error correction matter? |
| Off-the-shelf W4 | RedHatAI `Qwen3.5-{4B,9B}-quantized.w4a16`, `Qwen3-{8B,14B,32B}-quantized.w4a16`; `Qwen/Qwen3.5-27B-GPTQ-Int4`; `Qwen/Qwen3-{0.6B,1.7B}-GPTQ-Int8` | What people download (H5) |

**Default GPTQ recipe:** llm-compressor `GPTQModifier`, W4A16, group 128, symmetric, `actorder="static"`,
calibration 1,024 samples of Open-Platypus at up to 8,192 tokens (RedHat's recipe, so our Qwen3.5-4B/9B W4 can be
checked against theirs). Qwen3.5: quantize the MLPs and the 8 full-attention layers; keep the 24 Gated-DeltaNet
`linear_attn` layers, embeddings, `lm_head`, the vision tower and MTP heads in bf16, as RedHat does. (The official
`Qwen3.5-27B-GPTQ-Int4` is MLP-only, `-:.*attn.*`, a different recipe; it enters only as an off-the-shelf arm.)
Qwen3: every decoder Linear except `lm_head`. Calibration text must not overlap Fortress.

Off-the-shelf recipes differ from each other (MLP-only vs MLP + attention, Open-Platypus vs other data), which is
why the main comparison uses one recipe of our own across all sizes.

## Measurements (reuse the unified v2 pipeline unchanged)

- Fortress adversarial minus the 14 union exclusions, **rollouts 0-9 per question** (4,860 per condition), model-card
  thinking sampling (Qwen3 T=0.6; Qwen3.5 T=1.0, presence 1.5), reasoning cap 8,192, answer view 384 tokens, same
  per-rollout seeds and the same generation code as the baseline (ladder `one_pass` for Qwen3.5, `two_phase` for Qwen3).
- Judges as in v2: `vea_cal` (Qwen3.8-27B bf16, low-effort thinking, greedy) and F2 refusal (thinking off);
  category judge on VEA quotes; quote location check.
- Per condition: VEA rate, refusal rate, within-question gap (post's estimator), safety share of VEA, truncation
  rate, reasoning length, VEA per 1k reasoning tokens.
- Benign twins (2 rollouts per question): over-refusal.
- Capability: GSM8K (and MATH-500 for 9B+), to show the quantized model is still competent and to size the damage.

## Analysis

- Per model and condition: paired differences vs bf16 over the same questions, 95% CIs from 2,000 bootstrap draws
  resampling questions. Pooled: logistic mixed model, outcome ~ condition × model + (1 | question).
- Decompose Δrefusal = ΔVEA × gap + (change within VEA rollouts) + (change within non-VEA rollouts).
- Truncation: if it rises, report bounds treating truncated rollouts as all-refuse and none-refuse.
- Pre-registered primary outcomes: Δrefusal, ΔVEA, Δgap at W4, per model. Everything else is secondary.
- Hand-check: read 20 rollouts per condition for the pilot model (coherence, any new failure mode); re-check judge
  precision on 50 quantized-model VEA positives (the judge was calibrated on bf16 text).

## Phases

1. **Setup (Lonestar6).** Repo and venvs on `$WORK` (one vLLM venv as pinned here; a separate llm-compressor
   venv, since its torch/transformers pins may conflict). Check the A100 driver: the `+cu129` note in AGENTS.md
   applies only to old drivers. Weights and `HF_HOME` on `$SCRATCH`; our quantized checkpoints on `$WORK` (and
   pushed to private HF repos). Baseline rollouts: copy v2 `data/unified` if available.
2. **Pilot: Qwen3.5-4B** (VEA 15%, refusal 49%: both mid-range; small and cheap). Smoke first (20 questions × 4)
   per condition. Then full Fortress for bf16 rerun, W8, W4, W4 seed B, RTN W4, W3, RedHat W4. Deliver: rendered
   prompt per condition, one transcript per condition, stack-check result, GSM8K, and the H1-H5 table for one model.
   Decide W3 and the off-the-shelf arm from this.
3. **Scale:** W8 / W4 / (W3) for the other four Qwen3.5 sizes, then Qwen3 8B/14B/32B; seed-B W4 on 0.8B and 27B;
   off-the-shelf arms.
4. **Mechanism (only if Phases 2-3 show effects):** calibration-data ablation (generic chat vs the model's own
   benign reasoning traces vs a held-out refusal-heavy set, never Fortress); recipe ablation (MLP-only vs MLP +
   attention); teacher-forced log-likelihood of the bf16 model's VEA sentences under each quantized model vs
   matched non-VEA sentences (a low-variance test of whether quantization specifically disfavours VEA text);
   refusal-direction projection drift; the v2 injection / removal design on W4 for the model with the largest change.

## Cost (estimates to replace with pilot measurements)

Lonestar6 charges: `gpu-a100` 3 SU/node-hour (3× A100 40 GB), `gpu-a100-small` 1.5 SU/hour (1× A100), `gpu-h100`
6 SU/node-hour (2× H100 80 GB). Rough guesses, not measured:

- Generation: ~8-9M tokens per condition (4,860 rollouts × ~1.8k tokens). Small quantized models fit one A100, at
  very roughly 0.5-3 GPU-hours, i.e. 1-5 SU on `gpu-a100-small`. The bf16 27B rerun needs TP=2 (or an H100).
- Judging: Qwen3.8-27B bf16 (54 GB) needs TP=2 on A100-40GB or one H100. ~20M prefill + ~2.5M decode tokens per
  condition: very roughly 1-2 node-hours, 3-6 SU.
- Quantization: minutes to ~1 GPU-hour per checkpoint (27B layer-sequential on one A100).
- Pilot (7 conditions on 4B): roughly 50-80 SU. Full plan (~35-40 conditions): roughly 300-600 SU.
  DMS26009 has 5,822 SU; MLL has 142,900.

## Risks

- vLLM support for 3-bit compressed-tensors on Ampere, and for quantized Qwen3.5 checkpoints in the pinned vLLM 0.30.
- Judge drift: the Qwen3.8-27B judge was validated on bf16 text; quantized text may be stranger. Hence the re-check.
- Off-policy is not an issue here (no edits), but sampling differs by condition only through the weights; the seeds
  are matched per rollout, which does not couple samples across different weights.
- Harmful completions: rollouts stay gitignored and on TACC / the box only.
