#!/bin/bash
# Second queue: waits for the first queue, then fetches Qwen3.8-27B (subject + bf16 judge check).
# Qwen3.6-27B and Qwen3-32B are fetched later, after earlier subject weights are deleted (disk).
set -u
until grep -q "all queued downloads finished" /workspace/eval-awareness/log/exp02/downloads.log; do sleep 30; done
dl() { echo "[$(date +%H:%M:%S)] start $1@$2"; /venv/main/bin/hf download "$1" --revision "$2" --quiet >/dev/null && echo "[$(date +%H:%M:%S)] done  $1" || echo "[$(date +%H:%M:%S)] FAILED $1"; }
dl Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
echo "[$(date +%H:%M:%S)] queue 2 finished"
