#!/usr/bin/env bash
# exp02: time each judge engine variant (scripts/bench_judge_engines.py) in its own process.
#
#   scripts/bench_judge_engines.sh [variant ...]      # default: all variants
#
# Per-variant logs in log/bench/; the RESULT lines are collected in log/bench/exp02_judge_engines.log.
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}/scripts"
mkdir -p "${REPO_DIR}/log/bench"
SUMMARY="${REPO_DIR}/log/bench/exp02_judge_engines.log"
variants=("$@")
((${#variants[@]})) || variants=(fp8 bf16 fp8_triton_gdn fp8_no_prefix_cache)
for variant in "${variants[@]}"; do
  log="${REPO_DIR}/log/bench/exp02_judge_engine_${variant}.log"
  echo "[$(date -u +%H:%M:%S)] start ${variant}" >> "$SUMMARY"
  ./vllm_python.sh bench_judge_engines.py --variant "$variant" > "$log" 2>&1 \
    || echo "FAILED ${variant}; see ${log}" >> "$SUMMARY"
  grep "^RESULT" "$log" >> "$SUMMARY"
done
echo "[$(date -u +%H:%M:%S)] bench done" >> "$SUMMARY"
