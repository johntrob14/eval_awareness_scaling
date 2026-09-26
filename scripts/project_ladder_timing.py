"""Project exp01-style generation time for the Qwen3.5 dense ladder on A100 vs H100, bf16 vs FP8.

Run in /venv/main after the 4B workload benchmarks (scripts/bench_exp01_workload.py):
    python scripts/project_ladder_timing.py

Model of one vLLM decode step (decoding is memory-bandwidth bound here):
    step_s = overhead_s + bytes_moved / (efficiency * peak_bandwidth)
    bytes_moved = weights read per step
                + sum over running sequences of context_tokens * KV bytes per token
                + running sequences * 2 * linear-attention state bytes (read + write)
Running sequences are admitted first-come-first-served while their KV cache and
state fit in the GPU's cache budget; a sequence pushed out re-runs its prefill
when it comes back (compute-bound, charged at PREFILL_MFU of peak FLOPs).

(efficiency, overhead_s) are fitted per GPU on the measured 4B bf16 arms: the A100
run of exp01 (log/exp01/generate_*.log) and the H100 replays (log/bench/). The
H100 FP8 replays are held out to check the model before trusting projections.
The workload for every model is exp01's recorded 4B-base workload (same requests
and output lengths); other sizes will think for different lengths.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import exp01_config as cfg
import numpy as np
from huggingface_hub import get_safetensors_metadata, hf_hub_download

BENCH_DIR = cfg.REPO_ROOT / "log" / "bench"
WEIGHTS_CACHE = cfg.REPO_ROOT / "cache" / "bench" / "text_weight_bytes.json"
LADDER = ["0.8B", "2B", "4B", "9B", "27B"]
GIB = 2**30


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gib: float  # as reported by vLLM/torch
    peak_bandwidth: float  # bytes/s
    peak_dense_flops: dict[str, float]  # per weight precision


GPUS = {
    "A100-80GB": Gpu("A100-80GB", 79.25, 2.039e12, {"bf16": 312e12, "fp8": 312e12}),
    "H100-NVL": Gpu("H100-NVL", 93.1, 3.9e12, {"bf16": 835e12, "fp8": 1670e12}),
}
GPU_MEMORY_UTILIZATION = 0.85
NON_CACHE_OVERHEAD_GIB = (
    2.0  # activations, CUDA graphs, non-torch (vLLM log: ~1.7 GiB at 4B)
)
PREFILL_MFU = 0.4
MAX_NUM_SEQS = 256
CHUNK_STEPS = 64


# --- Model sizes ---------------------------------------------------------------


@dataclass(frozen=True)
class ModelShape:
    name: str
    n_params: int
    quantizable_bytes_bf16: int  # decoder-layer linear weights (FP8 halves these)
    other_bytes: int  # embeddings, lm_head, norms: stay bf16
    lm_head_bytes: int  # read in full every step (tied models: the embedding matrix)
    kv_bytes_per_token: int
    state_bytes_per_seq: int

    def resident_bytes(self, precision: str) -> int:
        linear = self.quantizable_bytes_bf16 // (2 if precision == "fp8" else 1)
        return linear + self.other_bytes

    def bytes_read_per_step(self, precision: str) -> int:
        linear = self.quantizable_bytes_bf16 // (2 if precision == "fp8" else 1)
        return linear + self.lm_head_bytes


def text_weight_bytes(repo_id: str) -> dict[str, int]:
    """Bytes of the text model's tensors by kind, from safetensors headers (no download)."""
    cache = json.loads(WEIGHTS_CACHE.read_text()) if WEIGHTS_CACHE.exists() else {}
    if repo_id not in cache:
        sizes = {"linear": 0, "embed": 0, "lm_head": 0, "other": 0, "n_params": 0}
        meta = get_safetensors_metadata(repo_id)
        for file_meta in meta.files_metadata.values():
            for name, tensor in file_meta.tensors.items():
                if "visual" in name or name.startswith("mtp"):
                    continue
                n = int(np.prod(tensor.shape))
                nbytes = n * {"BF16": 2, "F16": 2, "F32": 4}[tensor.dtype]
                if "embed_tokens" in name:
                    kind = "embed"
                elif name.startswith("lm_head"):
                    kind = "lm_head"
                elif (
                    ".layers." in name
                    and name.endswith(".weight")
                    and len(tensor.shape) == 2
                ):
                    kind = "linear"
                else:
                    kind = "other"
                sizes[kind] += nbytes
                sizes["n_params"] += n
        cache[repo_id] = sizes
        WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    return cache[repo_id]


def model_shape(size: str) -> ModelShape:
    repo_id = f"Qwen/Qwen3.5-{size}"
    config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
    text = config.get("text_config", config)
    layer_types = text["layer_types"]
    n_full = layer_types.count("full_attention")
    n_linear = layer_types.count("linear_attention")
    kv_per_token = n_full * 2 * text["num_key_value_heads"] * text["head_dim"] * 2
    state_per_seq = (
        n_linear
        * text["linear_num_value_heads"]
        * text["linear_key_head_dim"]
        * text["linear_value_head_dim"]
        * 2  # bf16 (vLLM's default for the gated-delta-net state)
    )
    sizes = text_weight_bytes(repo_id)
    tied = sizes["lm_head"] == 0
    return ModelShape(
        name=size,
        n_params=sizes["n_params"],
        quantizable_bytes_bf16=sizes["linear"],
        other_bytes=sizes["embed"] + sizes["lm_head"] + sizes["other"],
        lm_head_bytes=sizes["embed"] if tied else sizes["lm_head"],
        kv_bytes_per_token=kv_per_token,
        state_bytes_per_seq=state_per_seq,
    )


# --- Workload and simulation -----------------------------------------------------


def load_workload(variant: str) -> dict[str, list[tuple[int, int]]]:
    """Per arm, in submission order: (prompt tokens, generated tokens) per completion."""
    prompt_len = {
        p["sample_id"]: p["n_prompt_tokens"]
        for p in map(json.loads, cfg.PROMPTS_PATH.open())
    }
    workload: dict[str, list[tuple[int, int]]] = {arm: [] for arm in cfg.ARMS}
    for row in map(json.loads, cfg.GENERATIONS_PATH.open()):
        if row["model_variant"] == variant:
            workload[row["arm"]].append(
                (prompt_len[row["sample_id"]], row["n_generated_tokens"])
            )
    return workload


def cache_budget_bytes(gpu: Gpu, model: ModelShape, precision: str) -> float:
    usable = GPU_MEMORY_UTILIZATION * gpu.memory_gib * GIB
    return usable - model.resident_bytes(precision) - NON_CACHE_OVERHEAD_GIB * GIB


def simulate_arm(
    requests: list[tuple[int, int]],
    gpu: Gpu,
    model: ModelShape,
    precision: str,
    efficiency: float,
    overhead_s: float,
) -> float:
    """Wall seconds to generate one arm's requests, all submitted at once."""
    prompt = np.array([p for p, _ in requests], dtype=np.int64)
    target = np.array([n for _, n in requests], dtype=np.int64)
    done = np.zeros(len(requests), dtype=np.int64)
    was_running = np.zeros(len(requests), dtype=bool)
    budget = cache_budget_bytes(gpu, model, precision)
    bandwidth = efficiency * gpu.peak_bandwidth
    prefill_flops_per_token = 2 * model.n_params
    prefill_rate = PREFILL_MFU * gpu.peak_dense_flops[precision]
    wall = 0.0
    while (done < target).any():
        unfinished = np.flatnonzero(done < target)
        context = prompt[unfinished] + done[unfinished]
        need = (
            context + CHUNK_STEPS
        ) * model.kv_bytes_per_token + model.state_bytes_per_seq
        fits = np.cumsum(need) <= budget
        n_run = max(1, min(MAX_NUM_SEQS, int(fits.sum())))
        running = unfinished[:n_run]
        readmitted = running[~was_running[running] & (done[running] > 0)]
        wall += (
            (prompt[readmitted] + done[readmitted]).sum()
            * prefill_flops_per_token
            / prefill_rate
        )
        was_running[:] = False
        was_running[running] = True
        steps = int(min(CHUNK_STEPS, (target[running] - done[running]).min()))
        total_context = (prompt[running] + done[running]).sum() + len(
            running
        ) * steps / 2
        bytes_moved = (
            model.bytes_read_per_step(precision)
            + total_context * model.kv_bytes_per_token
            + len(running) * 2 * model.state_bytes_per_seq
        )
        wall += steps * (overhead_s + bytes_moved / bandwidth)
        done[running] += steps
    return wall


# --- Measurements and fitting -------------------------------------------------------


def a100_measured_arms() -> dict[tuple[str, str], float]:
    """(variant, arm) -> seconds, from the exp01 generation logs on the A100."""
    measured = {}
    pattern = re.compile(r"^(\w+) (\w+): \d+ completions in (\d+)s")
    for variant in cfg.MODEL_VARIANTS:
        for line in (cfg.LOG_DIR / f"generate_{variant}.log").read_text().splitlines():
            if match := pattern.match(line):
                measured[(match[1], match[2])] = float(match[3])
    return measured


def h100_measured_arms(precision: str) -> dict[tuple[str, str], float]:
    measured = {}
    for variant in cfg.MODEL_VARIANTS:
        report = json.loads(
            (BENCH_DIR / f"h100_4b_{precision}_{variant}-workload.json").read_text()
        )
        for arm, result in report["arms"].items():
            measured[(variant, arm)] = result["wall_s"]
    return measured


def predict_arms(gpu, model, precision, efficiency, overhead_s, workloads):
    return {
        (variant, arm): simulate_arm(
            requests, gpu, model, precision, efficiency, overhead_s
        )
        for variant, arms in workloads.items()
        for arm, requests in arms.items()
    }


EFFICIENCY_GRID = np.arange(0.30, 1.001, 0.05)
OVERHEAD_MS_GRID = np.arange(0.0, 20.01, 0.5)


def fit_gpu(gpu, model, measured, workloads) -> tuple[float, float]:
    """Grid search (efficiency, overhead) minimising squared log error over the measured arms."""
    best = None
    for efficiency, overhead_ms in itertools.product(EFFICIENCY_GRID, OVERHEAD_MS_GRID):
        predicted = predict_arms(
            gpu, model, "bf16", efficiency, overhead_ms / 1e3, workloads
        )
        error = sum(np.log(predicted[k] / measured[k]) ** 2 for k in measured)
        if best is None or error < best[0]:
            best = (error, float(efficiency), float(overhead_ms / 1e3))
    return best[1], best[2]


def main() -> None:
    shapes = {size: model_shape(size) for size in LADDER}
    workloads = {variant: load_workload(variant) for variant in cfg.MODEL_VARIANTS}
    four_b = shapes["4B"]

    fits, checks = {}, {}
    for gpu_name, measured in [
        ("A100-80GB", a100_measured_arms()),
        ("H100-NVL", h100_measured_arms("bf16")),
    ]:
        gpu = GPUS[gpu_name]
        fits[gpu_name] = fit_gpu(gpu, four_b, measured, workloads)
        predicted = predict_arms(gpu, four_b, "bf16", *fits[gpu_name], workloads)
        checks[f"{gpu_name} bf16 (fitted)"] = {
            f"{v}/{a}": {"measured_s": measured[(v, a)], "model_s": predicted[(v, a)]}
            for v, a in measured
        }
    measured_fp8 = h100_measured_arms("fp8")
    predicted_fp8 = predict_arms(
        GPUS["H100-NVL"], four_b, "fp8", *fits["H100-NVL"], workloads
    )
    checks["H100-NVL fp8 (held out)"] = {
        f"{v}/{a}": {
            "measured_s": measured_fp8[(v, a)],
            "model_s": predicted_fp8[(v, a)],
        }
        for v, a in measured_fp8
    }

    base_workload = {"base": workloads["base"]}
    projections = {}
    for size, (gpu_name, precision) in itertools.product(
        LADDER,
        [
            ("A100-80GB", "bf16"),
            ("A100-80GB", "fp8"),
            ("H100-NVL", "bf16"),
            ("H100-NVL", "fp8"),
        ],
    ):
        gpu, model = GPUS[gpu_name], shapes[size]
        budget = cache_budget_bytes(gpu, model, precision)
        key = f"{gpu_name} {precision}"
        if budget <= 0:
            projections.setdefault(size, {})[key] = {"fits": False}
            continue
        arms = predict_arms(gpu, model, precision, *fits[gpu_name], base_workload)
        projections.setdefault(size, {})[key] = {
            "fits": True,
            "cache_budget_gib": budget / GIB,
            "full_length_requests_that_fit": budget
            / (
                cfg.VLLM_MAX_MODEL_LEN * model.kv_bytes_per_token
                + model.state_bytes_per_seq
            ),
            "minutes": sum(arms.values()) / 60,
        }

    report = {
        "assumptions": {
            "workload": "exp01 4B-base requests and recorded output lengths, both arms (50 tasks x 2 conditions)",
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "kv_cache_dtype": "bf16",
            "fp8_on_a100": "weight-only FP8 (vLLM Marlin kernels); not measured",
            "prefill_mfu": PREFILL_MFU,
        },
        "fitted": {
            g: {
                "bandwidth_efficiency": e,
                "overhead_ms_per_step": o * 1e3,
                # An edge fit means the two-parameter step model does not describe that GPU well.
                "at_grid_edge": bool(
                    np.isclose(e, EFFICIENCY_GRID[[0, -1]]).any()
                    or np.isclose(o * 1e3, OVERHEAD_MS_GRID[[0, -1]]).any()
                ),
            }
            for g, (e, o) in fits.items()
        },
        "model_checks": checks,
        "model_shapes": {s: vars(m) for s, m in shapes.items()},
        "projected_minutes_per_model": projections,
    }
    out = BENCH_DIR / "ladder_projection.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("Fitted:", json.dumps(report["fitted"]))
    for name, rows in checks.items():
        print(f"\n{name}")
        for arm, r in rows.items():
            print(
                f"  {arm:28s} measured {r['measured_s']:7.1f}s  model {r['model_s']:7.1f}s"
            )
    columns = ["A100-80GB bf16", "A100-80GB fp8", "H100-NVL bf16", "H100-NVL fp8"]
    print("\nProjected minutes per model (exp01 workload)")
    print(f"{'model':6}" + "".join(f"{c:>17}" for c in columns))
    for size in LADDER:
        cells = []
        for c in columns:
            p = projections[size][c]
            cells.append("does not fit" if not p["fits"] else f"{p['minutes']:.0f}")
        print(f"{size:6}" + "".join(f"{c:>17}" for c in cells))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
