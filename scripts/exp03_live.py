"""Live preview of the exp03 figures while the overnight chain runs.

Run in /venv/main, in the background (it stops by itself after the chain ends):
    nohup /venv/main/bin/python scripts/exp03_live.py --wait-pid <pid of the chain> >> log/exp03/live.log 2>&1 &

Every POLL_SECONDS it checks which analysis inputs exist: the judge labels of each relabelled model, the logit v2
files, and the judged intervention answers. The chain writes each of these atomically (temporary file, then
rename), so a file that exists is complete. When that set changes, it re-runs exp03_analysis.py into a scratch
folder and swaps the result in as

    results/exp03_vea_ladder_v2/live/   and   figures/exp03_vea_ladder_v2/live/

Only these "live" folders (and their scratch copies) are ever replaced. The chain's own analysis runs ("final",
"final_with_qwen35_small_insertion") are separate folders and are never touched. The live numbers are provisional:
M1_length_standardized uses the CoT-length mix of the models included so far, so it shifts as models are added.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import exp03_analysis
import exp03_config as cfg
import exp03_generate
import exp03_judge

SCRIPTS_DIR = Path(__file__).resolve().parent
LIVE_NAME = "live"
SCRATCH_NAME = "live_next"
POLL_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def available_inputs() -> dict[str, list[str]]:
    """Models whose inputs exist, per input kind, using the same paths exp03_analysis.py reads."""
    decision = exp03_analysis.judge_decision()
    judged_interventions = []
    for m in cfg.INTERVENTION_MODELS + cfg.INTERVENTION_MODELS_ADDED:
        path = exp03_generate.intervention_path(m)
        if path.with_name(path.stem + "_callB.jsonl").exists():
            judged_interventions.append(m)
    return {
        "judge labels (F1-F4)": [m for m in cfg.RELABEL_MODELS if exp03_judge.label_path(m, decision).exists()],
        "judged interventions (F5)": judged_interventions,
        "logit v2 (F6)": [m for m in cfg.LOGIT_V2_MODELS if any(cfg.LOGIT_DIR.glob(f"{m}_*.jsonl"))],
    }


def status_markdown(inputs: dict[str, list[str]]) -> str:
    universe = {"judge labels (F1-F4)": cfg.RELABEL_MODELS,
                "judged interventions (F5)": cfg.INTERVENTION_MODELS + cfg.INTERVENTION_MODELS_ADDED,
                "logit v2 (F6)": cfg.LOGIT_V2_MODELS}
    lines = [f"# exp03 live preview, updated {utc_now()}", "",
             ("Provisional: re-rendered each time a model's inputs finish. The chain's own analysis ('final') is "
              "separate and is the one to cite."), ""]
    for kind, done in inputs.items():
        missing = [m for m in universe[kind] if m not in done]
        lines += [f"- **{kind}**: {len(done)} of {len(universe[kind])} models",
                  f"  - included: {', '.join(done) or 'none yet'}",
                  f"  - still to come: {', '.join(missing) or 'none'}"]
    return "\n".join(lines) + "\n"


def replace_dir(new: Path, live: Path) -> None:
    """Swap `new` in as `live` with two renames, so a reader never sees a half-written folder."""
    old = live.with_name(live.name + "_old")
    shutil.rmtree(old, ignore_errors=True)
    if live.exists():
        live.rename(old)
    new.rename(live)
    shutil.rmtree(old, ignore_errors=True)


def render(inputs: dict[str, list[str]]) -> bool:
    scratch_dirs = (cfg.RESULTS_DIR / SCRATCH_NAME, cfg.figure_dir(SCRATCH_NAME))
    for d in scratch_dirs:
        shutil.rmtree(d, ignore_errors=True)
    started = time.time()
    proc = subprocess.run(["nice", "-n", "19", sys.executable, "exp03_analysis.py", "--out-name", SCRATCH_NAME],
                          cwd=SCRIPTS_DIR, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"{utc_now()} analysis failed (exit {proc.returncode}); live folders left as they were\n"
              f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}", flush=True)
        return False
    status = status_markdown(inputs)
    for d in scratch_dirs:
        (d / "STATUS.md").write_text(status)
    replace_dir(cfg.RESULTS_DIR / SCRATCH_NAME, cfg.RESULTS_DIR / LIVE_NAME)
    replace_dir(cfg.figure_dir(SCRATCH_NAME), cfg.figure_dir(LIVE_NAME))
    print(f"{utc_now()} rendered in {time.time() - started:.0f} s: "
          + "; ".join(f"{k} {len(v)}" for k, v in inputs.items()), flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait-pid", type=int, required=True,
                        help="stop after the final render once this process (the chain) has exited")
    args = parser.parse_args()
    print(f"{utc_now()} watching; chain pid {args.wait_pid}", flush=True)
    rendered_inputs = None
    while True:
        chain_running = pid_alive(args.wait_pid)
        inputs = available_inputs()
        if inputs != rendered_inputs and any(inputs.values()):
            render(inputs)
            rendered_inputs = inputs  # after a failure, retry only when new inputs appear
        if not chain_running:
            print(f"{utc_now()} chain pid {args.wait_pid} has exited; stopping", flush=True)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
