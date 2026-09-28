# AGENTS.md (on top of existing .md)

Instance/ops guide (Vast): /workspace/AGENTS.md; this file adds project rules on top.

## STATE (human-maintained; update at every close)
Project: Which small open models show verbalized eval awareness (VEA) that actually changes behaviour.
Question (falsifiable sentence):
So what:
Established so far (exp ids):
In flight (exp ids): exp01_organism_gap (run complete, UNVERIFIED; awaiting human check)
Current story candidate:
Next experiment and why: Today: positive control on the Qwen3.5-4B eval-aware organism, then a VEA pilot on Qwen3-8B.
Kill criteria in force:
Last updated: 2026-09-25 by agent (exp01 run complete)

## Roles
- Agent: implement, run, report in the format below.
- Humans: design experiments, choose baselines and controls, read raw data, recompute headlines, interpret.
  If you think a design is wrong, say so, propose an alternative, and wait.

## Environment
- Persistent Jupyter kernel via the `jupyter` MCP server (fallback: ipython in tmux session `py`).
- Load models/data in dedicated cells at the top. Never restart the kernel or reload a loaded model without asking.
- Save every plot to figures/<exp_id>_<name>.png as well as displaying it.
- Jobs longer than 10 minutes run as background scripts with logs, not notebook cells.
- Large-scale sampling uses vLLM (venv at /workspace/.venv-vllm, always run via scripts/vllm_python.sh, which adds CUDA forward-compat libs on older drivers); activations use nnsight/HF in the main venv.
- New instance: restore envs exactly from log/setup/freeze_*.txt and rebuild models with `exp01_build_models.py --restore-report` (see log/restore_h100/).
- Use uv for packages. Never pip install into system Python.

## Experiments
- Every experiment: id expNN_name, directory results/expNN_name/, manifest.json written BEFORE the run.
- Never overwrite or hand-edit results. Never change metric code between a baseline and its comparison without re-running the baseline.
- Always run the control the humans specified. If none was specified, stop and ask.
- Report n, seeds, bootstrap CI, and the baseline next to every headline number.
- Cache expensive intermediates in cache/ with content-addressed names.

## Report format (end of every experiment)
EXP <id> @ <commit> | model <id> | n=<n> | seeds=<list>
Claim tested:
Headline: <metric> = <value> [CI]   Baseline: <value> [CI]   Control: <value> [CI]
Figures:
Three dumbest ways this could be wrong: 1) ... checked? 2) ... checked? 3) ... checked?
For the human to verify: <file, rows, number to recompute>
Status: UNVERIFIED until a human adds it to VERIFIED.md

## Evidence norms
- Recomputed rates/AUCs > plotted curves > judge scores. Label judge scores "judge", with human agreement if known.
- Random examples by fixed seed, stated as such.
- Interventions at matched norm/dose; report collateral damage (incoherence, capability) alongside the effect.
- A null with a working control is a result.

## Git and writing
- Commit when a figure exists (exp id in the message) and again when verified. No force-push; no amending pushed commits.
- Drafts only. Never state a result as established unless it is in VERIFIED.md. No filler.
