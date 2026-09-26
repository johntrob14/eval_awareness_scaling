"""Inference backend selection: MLX on Apple Silicon, vLLM on NVIDIA/CUDA."""

import platform
import sys


def default_backend() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "vllm"


def load_vllm(model: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
              max_model_len: int | None = None, **kwargs):
    from vllm import LLM

    return LLM(model=model, tensor_parallel_size=tensor_parallel_size,
               gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, **kwargs)


def add_vllm_args(ap) -> None:
    ap.add_argument("--tensor-parallel-size", "--tp", type=int, default=1, help="vLLM: GPUs per model")
    ap.add_argument("--pipeline-parallel-size", "--pp", type=int, default=1,
                    help="vLLM: pipeline stages (cheaper than TP when GPUs lack NVLink)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM")
    ap.add_argument("--max-model-len", type=int, help="vLLM: context length (default: model's)")
