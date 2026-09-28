# exp03 run history

Chain scripts from the exp03 run that were restarted or replaced mid-run. They are the record of what ran, not
entry points: each `cd`s to its own directory and calls sibling scripts, so they do not run from here.

| script | role in the run |
|---|---|
| `exp03_full_chain_resume.sh` | first restart of `scripts/exp03_full_chain.sh` (after the Qwen3.5-9B judge smoke test) |
| `exp03_full_chain_resume2.sh` | second restart (after a vLLM teardown abort); stopped before the judge steps |
| `exp03_chain_per_model.sh` | per-model order; relabeled Qwen3.8-27B, then was replaced by `scripts/exp03_chain_logit_last.sh` |
| `exp03_insertion_extension.sh`, `exp03_qwen3_32b_ladder.sh`, `exp03_attach_followers.sh` | followers of the main chain; replaced by `scripts/exp03_chain_logit_last.sh` before they did any work |

Step-by-step timestamps of the whole run are in `log/exp03/full_chain.log` (not committed; see the top-level README).
