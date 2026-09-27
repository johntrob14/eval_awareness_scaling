#!/bin/bash
# Attach the two followers to a running exp03 main chain, given its PID:
#   1. a waiter that starts exp03_insertion_extension.sh when the chain exits (the extension checks "== done"),
#   2. the live-figure watcher (exp03_live.py), which stops after the waiter (and so the extension) has exited.
# Usage: bash scripts/exp03_attach_followers.sh <chain pid>
set -euo pipefail
cd "$(dirname "$0")/.."
CHAIN=$1
kill -0 "$CHAIN"
setsid nohup bash -c "while kill -0 $CHAIN 2>/dev/null; do sleep 60; done; bash scripts/exp03_insertion_extension.sh" \
    >> log/exp03/insertion_extension.log 2>&1 < /dev/null &
WAITER=$!
setsid nohup /venv/main/bin/python scripts/exp03_live.py --wait-pid "$WAITER" >> log/exp03/live.log 2>&1 < /dev/null &
echo "chain $CHAIN, waiter $WAITER, watcher $!"
