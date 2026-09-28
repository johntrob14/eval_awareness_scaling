# Refusal judge comparison

240 random finished adversarial answers, 60 each from Gemma 4 31B, Nemotron 3 Super, Qwen3.5-122B, Qwen3.5-397B (seed 20260929). Both judges are Qwen3.8-27B. Each judge sees two views of every answer: the first 384 tokens (the collaborator's cap, subject tokenizer) and the first 6,000 characters (mine).

- mine: the post's F2, binary, greedy, thinking off (bf16)
- theirs: exp03 3-way (full_refusal / partial / comply), T=0.7, FP8, majority of 3 runs; binary = full_refusal

## Refusal rate

| model | answers >384 tok | mine, 6k chars (my pipeline) | mine, 384 tok | theirs, 6k chars | theirs, 384 tok (their pipeline) |
|---|---|---|---|---|---|
| gemma-4-31B-it | 53% | 53% | 55% | 48% | 50% |
| Nemotron-3-Super | 90% | 52% | 53% | 42% | 53% |
| Qwen3.5-122B | 78% | 58% | 55% | 43% | 47% |
| Qwen3.5-397B | 63% | 70% | 78% | 55% | 57% |
| all | 71% | 58% | 60% | 47% | 52% |

## Agreement (binary)

- as deployed (mine 6k vs theirs 384): 90% (n=240)
- rubric only, same 6k input: 88% (n=240)
- rubric only, same 384 input: 89% (n=240)
- input only, my rubric (6k vs 384): 96% (n=240)
- input only, their rubric (6k vs 384): 94% (n=240)
- their judge, same label in all 3 runs: 6k 89%, 384 89%

## Their 3-way label vs my binary (same 6k input)

| theirs \ mine | refusal | not refusal |
|---|---|---|
| full_refusal | 112 | 1 |
| partial | 28 | 16 |
| comply | 0 | 83 |
| None | 0 | 0 |

As-deployed disagreements: 24 (mine-only refusal 20, theirs-only 4). Sheet: `disagreements.md` / `disagreements_labels.csv`.
