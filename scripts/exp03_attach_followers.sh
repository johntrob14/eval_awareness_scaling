#!/bin/bash
# Attach the followers to a running exp03 main chain, given its PID. Each starts when the previous one exits:
#   1. exp03_insertion_extension.sh (checks the main chain's "== done"),
#   2. exp03_qwen3_32b_ladder.sh (Qwen3-32B's rollouts and labels, then the final analysis; also checks "== done"),
#   3. the live-figure watcher (exp03_live.py) runs throughout and stops after follower 2 has exited.
# Usage: bash scripts/exp03_attach_followers.sh <chain pid>
set -euo pipefail
cd "$(dirname "$0")/.."
CHAIN=$1
kill -0 "$CHAIN"
setsid nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 60; done; bash scripts/exp03_insertion_extension.sh" \
    >> log/exp03/insertion_extension.log 2>&1 < /dev/null &
EXTENSION=$!
setsid nohup bash -c "while kill -0 $EXTENSION 2>/dev/null; do sleep 60; done; bash scripts/exp03_qwen3_32b_ladder.sh" \
    >> log/exp03/qwen3_32b_ladder.log 2>&1 < /dev/null &
QWEN3_32B=$!
setsid nohup /venv/main/bin/python scripts/exp03_live.py --wait-pid "$QWEN3_32B" >> log/exp03/live.log 2>&1 < /dev/null &
echo "chain $CHAIN, extension waiter $EXTENSION, Qwen3-32B waiter $QWEN3_32B, watcher $!"
