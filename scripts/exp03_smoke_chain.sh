#!/bin/bash
# exp03 smoke tests, in order on the one GPU. Each step skips work whose outputs exist.
#   1. judge: FP8 pass (bf16 passes run separately first)   2. MTP off / on, Qwen3.8-27B
#   3. reports -> decision files (judge engine, MTP)        4. intervention splice run on Qwen3.8-27B (3 bases)
set -euo pipefail
cd "$(dirname "$0")"
VLLM=./vllm_python.sh
PY=/venv/main/bin/python
LOG=../log/exp03

while pgrep -f "exp03_judge.py smoke --engine bf16" > /dev/null; do sleep 30; done

$VLLM exp03_judge.py smoke --engine bf16 >> $LOG/smoke_judge_bf16.log 2>&1  # no-op unless a pass is missing
$VLLM exp03_judge.py smoke --engine fp8 > $LOG/smoke_judge_fp8.log 2>&1
$VLLM exp03_generate.py mtp-smoke --mtp-tokens 0 > $LOG/smoke_mtp0.log 2>&1
$VLLM exp03_generate.py mtp-smoke --mtp-tokens 2 > $LOG/smoke_mtp2.log 2>&1
$PY exp03_smoke_report.py judge > $LOG/smoke_report_judge.log 2>&1
$PY exp03_smoke_report.py mtp > $LOG/smoke_report_mtp.log 2>&1
$VLLM exp03_generate.py interventions --models Qwen3.8-27B --limit-bases 3 > $LOG/smoke_interventions.log 2>&1
$VLLM exp03_judge.py interventions --models Qwen3.8-27B --smoke-bases 3 > $LOG/smoke_interventions_judge.log 2>&1
$PY exp03_smoke_report.py interventions --model Qwen3.8-27B --n-bases 3 > $LOG/smoke_report_interventions.log 2>&1
echo "smoke chain done $(date)"
