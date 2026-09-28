#!/bin/bash
# Start an exp03 chain script detached, plus the live-figure watcher, which stops after the chain has exited.
# Usage: bash scripts/exp03_launch.sh scripts/exp03_chain_logit_last.sh
set -euo pipefail
cd "$(dirname "$0")/.."
setsid nohup bash "$1" >> log/exp03/full_chain.log 2>&1 < /dev/null &
CHAIN=$!
setsid nohup /venv/main/bin/python scripts/exp03_live.py --wait-pid "$CHAIN" >> log/exp03/live.log 2>&1 < /dev/null &
echo "chain $CHAIN, watcher $!"
