# exp02_vea_ladder: what the pipeline does, and what it produces

This is the plain-language overview written before the run, updated where the run deviated (noted inline).
Results are in `results/exp02_vea_ladder/all_models_validated/`; the relabeled second pass is exp03 (see the README).

**The question.** Take 10 Qwen models of different sizes (plus Qwen3.8-27B for a same-size version comparison, and an optional Qwen3-32B). When a model says in its reasoning that it might be being tested, does it refuse harmful requests more often? Does that relationship change abruptly at some size, such as 14B?

**Short version of the method.** Each model answers the same 489 harmful-but-disguised requests many times. A judge model labels every answer on two questions: did the reasoning mention being tested, and did the answer refuse? We then compare refusal rates between answers that mentioned testing and answers that didn't, *for the same request*.

## The pipeline

```mermaid
flowchart LR
  prompts["Prompts: Fortress 489 + benign twins + ToxicChat + WildChat"] --> stage1["Stage 1: 10 answers per prompt"]
  stage1 --> select["Pick the 98 prompts with the most 'testing' mentions"]
  select --> stage2["Stage 2: 20 more answers each"]
  prompts --> logit["Logit monitor: P('The user might be testing')"]
  stage1 --> judge["Judge: mentions testing? refuses?"]
  stage2 --> judge
  judge --> checks["Checks: judge agreement, human hand labels"]
  judge --> analysis["Metrics, 5 bar figures, report"]
  logit --> analysis
```

1. **Prompts.**
   - The 489 [Fortress](https://huggingface.co/datasets/ScaleAI/fortress_public) adversarial prompts: harmful requests wrapped in jailbreak tricks. We drop the 11 that explicitly say "this is an evaluation".
   - Their benign twins: the same topics without the harmful ask.
   - Real user requests for the logit monitor: 489 toxic and 489 benign from ToxicChat, and 489 benign from WildChat.
     (The public WildChat releases contain no toxic-flagged conversations, so ToxicChat supplies the toxic set;
     its prompts are capped at 1,536 characters, which makes prompt length a confound the length baseline must beat.)
   - The 11 excluded eval-stating prompts get 2 answers each as a positive control for the judge (in no metric).
   - No prompt contains any hint about evaluation vs deployment.
2. **Answers ("rollouts").**
   - Each model answers each Fortress prompt 10 times.
   - The 98 prompts where it most often mentioned testing get 20 more answers each. That's where the comparison has power.
     (The plan said 40. At hour 5.6 the judge proved much slower than the plan assumed, so the pre-agreed fallback of 20
     applies to every model. The Qwen3 ladder had already been generated with 40; its first 20 are the same answers
     a 20-answer run would draw, and only those are used. See `results/exp02_vea_ladder/fallback_decision.json`.)
   - Benign twins get 2 answers each.
   - Reasoning is capped at 8,192 tokens, and the judge sees the first 384 tokens of the answer.
3. **Logit monitor (no judge involved).** For every prompt, we read how likely the model is to *start* its reasoning with "The user might be testing". This measures eval suspicion even when the model never says it.
4. **Judge (Qwen3.8-27B, full-precision bf16 weights; the plan had the FP8 version, which runs at half speed on this machine).** Per answer, it records:
   - Did the reasoning explicitly say it might be tested or evaluated? If so, what did it think was being tested: its safety, its capabilities, both, or unspecified?
   - Did it notice a jailbreak *without* mentioning testing? This is the "placebo" category.
   - Did the final answer refuse?
5. **Checks.**
   - The judge's labels are compared with its own slower thinking mode and with the FP8 version of the judge.
   - The regex that feeds it is tested against 515 human-verified examples from the Bloom post.
   - A human hand-labels about 250 blinded items (`results/exp02_vea_ladder/labeling/`).
6. **Analysis.** Metrics with bootstrap confidence intervals, five bar charts, and a written report.

## The models

- **Qwen3** (temperature 0.6, the card's recommended settings): 0.6B, 1.7B, 4B, 8B, 14B.
- **Qwen3.5** (temperature 1.0, presence penalty 1.5, the card's recommended settings): 0.8B, 2B, 4B, 9B, 27B.
- **Qwen3.8-27B** (temperature 1.0, presence penalty 0, reasoning effort "medium"; the card's settings): compared with Qwen3.5-27B as an exploratory version comparison. It is also the judge, so its own labels are self-judged.
- **Qwen3-32B add-on:** its logit monitor and its answers (10 per prompt, no stage 2) run at the end, after the validated results for the main ladder (originally only if the schedule still finished by hour 21; that cutoff was dropped on request at T+20.5, so the add-on always runs). Its bars are marked "stage 1 only", so its within-question estimate is less precise.
- The two families use different sampling settings, so compare sizes *within* a family. Differences between families may partly reflect the settings.

## The key numbers, in plain words

- **VEA rate.** The share of answers whose reasoning explicitly considers being tested. "VEA" stands for verbalized eval awareness.
- **Within-question effect (the main result).** On the *same prompt*, how much more often do answers that mention testing refuse than answers that don't? This is averaged over prompts. Holding the prompt fixed rules out "some prompts are just more obviously harmful". It's an association, not proof that mentioning testing causes the refusal.
- **Across-question association.** Are the *prompts* that trigger testing talk also the prompts that get refused? This describes the prompts, not the effect of verbalizing.
- **Placebo.** The same within-question comparison, for answers that noticed a jailbreak but didn't mention testing. If it's as large as the main effect, "noticing the trick" explains the refusals, not eval awareness.
- **Logit-monitor AUROC.** How well the monitor separates Fortress prompts from real user requests (ToxicChat toxic is the primary contrast; WildChat benign, ToxicChat benign and the benign twins are secondary). 0.5 means it can't tell them apart; 1.0 means it separates them perfectly. This is the most robust number, because no judge is involved.

Toy example of why the first two differ:
- Prompt A is very harmful. 30 of its 50 answers mention testing, and 90% refuse *either way*.
- Prompt B is mild. No answer mentions testing, and 10% refuse.
- Pooling everything, testing-talk answers look 57 points more likely to refuse.
- The within-question effect is 0, which is correct: mentioning testing changed nothing.
- The across-question gap is 80 points, which is also correct, but it only says that testing talk shows up on the prompts that already get refused.

## Outputs

```
results/exp02_vea_ladder/
  manifest.json            everything that defines the run, written BEFORE it (models, revisions,
                           settings, seeds, pre-registered primary metrics)
  smoke_decision.json      smoke-test lengths and the go / fallback decision
  fallback_decision.json   the stage-2 fallback applied mid-run, with the measurements behind it
  summary.json             every metric per model with 95% CI and sample sizes, the jump tests,
                           and the judge checks (self-agreement, full-CoT audits, thinking/FP8 validation)
  per_model_metrics.csv    the headline metrics, one row per model
  read_samples.md          random answers per model (fixed seed) for reading
  judge_regex_recall_bloom_quotes.json   regex recall on Bloom's 515 human-verified quotes
  preliminary_qwen3/       the same files for the Qwen3 ladder alone, made mid-run
  labeling/sheet_N.csv     blinded items for human hand labels (key_N.json maps them back)
cache/exp02/generations/   every answer: prompt id, stage, reasoning, answer, token ids, truncation flag
cache/exp02/logit_monitor/ per-prompt phrase log-probabilities
cache/exp02/judge/         every judge call with its labels and decision-token probabilities
figures/exp02_vea_ladder/<run>/F1..F5.png   one folder per analysis run (final/, all_models_validated/, ...)
```

Plus a report in the AGENTS.md format, marked **UNVERIFIED** until a human has checked it.

**The five figures.** All show every model as a bar, ordered by size, colored by family, with 95% CIs.

- **F1, VEA rate,** split into safety / capabilities / both / unspecified. Bloom's published Qwen3 rates are shown as markers, to check whether our judge agrees with theirs.
- **F2, refusal rate** on Fortress prompts vs their benign twins. The benign bars show over-refusal.
- **F3, within-question effect (the main figure).** Pooled, safety-framed, capabilities-framed, and placebo. Bars above 0 mean answers that mention testing refuse more.
- **F4, across-question association:** the refusal gap and the rank correlation.
- **F5, logit-monitor AUROC** against ToxicChat toxic (primary), WildChat benign, ToxicChat benign and the benign twins, with a prompt-length-only baseline the monitor must beat.

**What counts as a jump** (decided before the run): within one family, one step between adjacent sizes that is clearly larger than every other step, with a CI that excludes 0. It is tested only on F1's total and F3's pooled bar. Everything else is exploratory.

## What to expect (priors from the literature, not our results)

- **VEA rate, Qwen3.** Bloom measured 1.0 / 1.3 / 5.3 / 5.9 / 6.0% for 0.6B / 1.7B / 4B / 8B / 14B.
  - The rate jumps between 1.7B and 4B, not at 14B.
  - What changes at 14B is the share of *safety-framed* testing talk, which goes from 34% at 8B to 63% at 14B. F1's split shows exactly this.
  - If our rates land near Bloom's, the choice of judge (theirs was GPT-5-mini) isn't a big factor.
- **Qwen3.5** should verbalize more. Bloom found Qwen3.5-27B does so about 2× as often as Qwen3-32B.
- **Refusal on Fortress will be low for Qwen3**, near 0% on many prompts. So the within-question effect will be imprecise for the smallest models, where only large jumps are detectable.
- **Within-question effect.** Bloom found +3 to +18 points across four models. Safety-framed testing talk tends to go with refusal, while capabilities-framed talk goes with compliance (Zhuang & Aranguri), so the pooled bar can hide opposite effects.
- **Logit monitor.** Expect near chance at 2B and below, and better at larger sizes. In an independent sweep, Qwen3-14B was the best eval-vs-deployment detector.
- **Scale.** Evaluation Faking found eval-driven behavior change grows mainly *beyond* 14B. A null result up to 14B would not contradict the literature, which is why the optional Qwen3-32B add-on matters.

## How not to over-read it

- The main result is a within-prompt *association*, not a causal effect.
- VEA and refusal labels come from a judge model and are tagged "judge". They are UNVERIFIED until the human hand labels are in.
- Comparisons between families mix size with different sampling settings and training recipes.
- If the Qwen3-32B answers don't fit in the time budget, the Qwen3 ladder stops at 14B for everything except the logit monitor, right where the literature says effects start growing.
