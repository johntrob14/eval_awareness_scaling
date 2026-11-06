# Sourced by the quantization-study job scripts on Lonestar6.
set -euo pipefail
cd "$WORK/eval_awareness_scaling"
git pull -q --ff-only origin quantization || echo "warning: could not update the checkout; running $(git rev-parse --short HEAD)"
. .sauce/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1  # everything is downloaded on the login node first
mkdir -p "$QUANT_DATA/logs"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is); code $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
