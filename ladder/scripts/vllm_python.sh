#!/usr/bin/env bash
# Python of the vLLM venv; use this instead of /workspace/.venv-vllm/bin/python.
# The venv's torch is built for CUDA 13.2, which needs driver >= 595. On an older
# data-center driver (e.g. 570 on the H100 host), NVIDIA's forward-compat libcuda
# (apt package cuda-compat-13-2) goes first on the library path for this process
# tree only; /venv/main keeps using the host driver.
set -euo pipefail
VLLM_VENV=/workspace/.venv-vllm
COMPAT_DIR=/usr/local/cuda-13.2/compat
MIN_NATIVE_DRIVER_MAJOR=595

driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if ((driver_major < MIN_NATIVE_DRIVER_MAJOR)); then
  if [[ ! -e "${COMPAT_DIR}/libcuda.so.1" ]]; then
    echo "Driver ${driver_major} < ${MIN_NATIVE_DRIVER_MAJOR} and ${COMPAT_DIR} is missing:" \
      "apt-get install cuda-compat-13-2" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
exec "${VLLM_VENV}/bin/python" "$@"
