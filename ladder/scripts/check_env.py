"""Print the project's hardware and software environment.

Run with the research env:  /venv/main/bin/python scripts/check_env.py
The vLLM version is read from its own venv, since vLLM pins a different torch.
"""

import importlib.metadata
import subprocess
from pathlib import Path

import torch

VLLM_PYTHON = str(Path(__file__).with_name("vllm_python.sh"))
MAIN_ENV_PACKAGES = ["transformers", "peft", "nnterp"]

VLLM_PROBE = """
import torch, vllm
x = torch.ones(64, 64, device="cuda")
print(f"vllm            {vllm.__version__}")
print(f"torch           {torch.__version__}")
print(f"torch CUDA      {torch.version.cuda}")
print(f"GPU op ok       {bool((x @ x).sum().item() == 64**3)}")
"""


def print_gpu() -> None:
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print("== GPU ==")
    print(f"name            {props.name}")
    print(f"memory          {props.total_memory / 2**30:.1f} GiB")
    print(f"driver          {driver}")


def print_main_env() -> None:
    x = torch.ones(64, 64, device="cuda")
    print("== Research env (/venv/main) ==")
    print(f"torch           {torch.__version__}")
    print(f"torch CUDA      {torch.version.cuda}")
    print(f"GPU op ok       {bool((x @ x).sum().item() == 64**3)}")
    for name in MAIN_ENV_PACKAGES:
        print(f"{name:<15} {importlib.metadata.version(name)}")


def print_vllm_env() -> None:
    print(f"== Generation env ({VLLM_PYTHON}) ==")
    result = subprocess.run(
        [VLLM_PYTHON, "-c", VLLM_PROBE], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}):\n{result.stderr}")
        return
    print(result.stdout, end="")


if __name__ == "__main__":
    print_gpu()
    print_main_env()
    print_vllm_env()
