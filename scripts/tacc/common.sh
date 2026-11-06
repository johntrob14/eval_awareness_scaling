# Sourced by the quantization-study job scripts on Lonestar6.
# TACC rule: compute nodes touch $WORK only to stage data in or out. Job I/O therefore goes to a live data
# directory on $SCRATCH (copied from the $WORK archive the first time), and every job copies it back to the
# archive when it exits (rsync --update: an older copy never replaces a newer one). Weights are read from
# $SCRATCH (HF cache) and our checkpoints from $WORK once at engine load (staging in).
set -euo pipefail
cd "$WORK/eval_awareness_scaling"
git pull -q --ff-only origin quantization || echo "warning: could not update the checkout; running $(git rev-parse --short HEAD)"
. .sauce/env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1  # downloads happen in their own jobs
export QUANT_DATA_ARCHIVE=$WORK/results/eval_awareness_scaling/data
export QUANT_DATA=$SCRATCH/eval_awareness_scaling/data
JOBLOGS=$SCRATCH/eval_awareness_scaling/logs
mkdir -p "$QUANT_DATA/logs" "$JOBLOGS"
# Stage in only what $SCRATCH lacks (first job, or after a purge); never overwrite a live file another job appends to.
rsync -a --ignore-existing "$QUANT_DATA_ARCHIVE/" "$QUANT_DATA/"
stage_out() {
  rsync -a --update "$QUANT_DATA/" "$QUANT_DATA_ARCHIVE/" &&
  rsync -a --update "$JOBLOGS/" "$WORK/results/eval_awareness_scaling/logs/" &&
  echo "staged out to $QUANT_DATA_ARCHIVE at $(date -Is)"
}
trap stage_out EXIT
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is); code $(git rev-parse --short HEAD); data $QUANT_DATA"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
