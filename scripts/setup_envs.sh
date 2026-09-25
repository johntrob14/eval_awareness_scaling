#!/usr/bin/env bash
# Builds the two Python environments for this project. Stops at the first failure.
#   Research env:   /venv/main (Vast image; its CUDA-matched torch stack must not change)
#   Generation env: /workspace/.venv-vllm (vLLM pins its own torch, so it lives apart)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_PY=/venv/main/bin/python
VLLM_VENV=/workspace/.venv-vllm
FREEZE_DIR="${REPO_DIR}/log/setup"

RESEARCH_PACKAGES=(
  transformers accelerate peft datasets pandas pyarrow scipy matplotlib openai nnterp jupytext ruff
)
TORCH_FAMILY_REGEX='^(torch|torchvision|torchaudio|torchcodec|triton|nvidia-[a-z0-9-]+)=='

torch_family_versions() {
  uv pip freeze --python "$MAIN_PY" | grep -iE "$TORCH_FAMILY_REGEX"
}

mkdir -p "$FREEZE_DIR"

echo "== Research env (${MAIN_PY}) =="
pins_file="$(mktemp)"
trap 'rm -f "$pins_file"' EXIT
torch_family_versions > "$pins_file"
echo "Pinning the image's torch stack as constraints:"
cat "$pins_file"

uv pip install --python "$MAIN_PY" --constraints "$pins_file" "${RESEARCH_PACKAGES[@]}"

if ! diff <(cat "$pins_file") <(torch_family_versions); then
  echo "ERROR: the torch stack in /venv/main changed during install (diff above)." >&2
  exit 1
fi
echo "Torch stack unchanged: $(grep -i '^torch==' "$pins_file")"

echo "== Generation env (${VLLM_VENV}) =="
if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
  uv venv "$VLLM_VENV" --python 3.12 --managed-python --seed
fi
uv pip install --python "${VLLM_VENV}/bin/python" vllm --torch-backend=auto

uv pip freeze --python "$MAIN_PY" > "${FREEZE_DIR}/freeze_main.txt"
uv pip freeze --python "${VLLM_VENV}/bin/python" > "${FREEZE_DIR}/freeze_vllm.txt"
echo "Wrote ${FREEZE_DIR}/freeze_main.txt and ${FREEZE_DIR}/freeze_vllm.txt"
