"""Compare quantized checkpoints weight by weight (compressed-tensors pack-quantized W4A16 / W8A16).

    uv run python -m eval_awareness.quant.compare_weights --base Qwen3.5-4B \
        --variants gptq-w4 hf-w4 gptq-w4-calB rtn-w4 --out compare.json

For every quantized Linear present in all checkpoints: the fraction of integer weights identical between each pair
of checkpoints, the median relative difference of their group scales, and each checkpoint's relative
reconstruction error ||W_deq - W_bf16|| / ||W_bf16|| against the original bf16 weights. The bf16 error doubles as a
check of the unpacking: a wrong unpack gives an error near or above 1, a right one a few percent.
Packing: signed integers stored as unsigned (q + 2^(b-1)), packed little-endian into int32 (32/b values each);
symmetric, one scale per group of `group_size` input columns.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_awareness.quant import config as qc


class Checkpoint:
    def __init__(self, path: Path):
        self.path = Path(path)
        cfg = json.loads((self.path / "config.json").read_text())
        q = cfg.get("quantization_config") or {}
        self.bits, self.group = None, None
        if q.get("config_groups"):
            w = next(iter(q["config_groups"].values()))["weights"]
            self.bits, self.group = w["num_bits"], w["group_size"]
        self.where = {}
        for f in self.path.glob("*.safetensors"):
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    self.where[k] = f

    def tensor(self, key: str) -> torch.Tensor:
        with safe_open(str(self.where[key]), "pt") as h:
            return h.get_tensor(key)

    def quantized_layers(self) -> set[str]:
        return {k.removesuffix(".weight_packed") for k in self.where if k.endswith(".weight_packed")}

    def ints(self, layer: str) -> torch.Tensor:
        packed = self.tensor(layer + ".weight_packed").to(torch.int64) & 0xFFFFFFFF
        per = 32 // self.bits
        shifts = torch.arange(per, dtype=torch.int64) * self.bits
        vals = (packed.unsqueeze(-1) >> shifts) & ((1 << self.bits) - 1)
        out, cols = self.tensor(layer + ".weight_shape").tolist()
        return (vals.reshape(packed.shape[0], -1)[:, :cols] - (1 << (self.bits - 1))).reshape(out, cols)

    def scales(self, layer: str) -> torch.Tensor:
        return self.tensor(layer + ".weight_scale").float()

    def dequant(self, layer: str) -> torch.Tensor:
        q, s = self.ints(layer).float(), self.scales(layer)
        return (q.reshape(q.shape[0], -1, self.group) * s.unsqueeze(-1)).reshape(q.shape)


def bf16_weight(base: Path, where: dict, layer: str) -> torch.Tensor:
    with safe_open(str(where[layer + ".weight"]), "pt") as h:
        return h.get_tensor(layer + ".weight").float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cks = {v: Checkpoint(qc.weights(f"{args.base}@{v}")["path"]) for v in args.variants}
    for v, c in cks.items():
        print(f"{v}: {c.bits}-bit, group {c.group}, {len(c.quantized_layers())} quantized layers, {c.path}", flush=True)
    bf16 = Path(qc.weights(f"{args.base}@bf16")["path"])
    bf16_where = {}
    for f in bf16.glob("*.safetensors"):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                bf16_where[k] = f
    layers = sorted(set.intersection(*(c.quantized_layers() for c in cks.values())))
    print(f"{len(layers)} layers quantized in every checkpoint", flush=True)

    rows = []
    for layer in layers:
        w = bf16_weight(bf16, bf16_where, layer) if layer + ".weight" in bf16_where else None
        ints = {v: c.ints(layer) for v, c in cks.items()}
        row = {"layer": layer, "n_weights": int(next(iter(ints.values())).numel())}
        for v, c in cks.items():
            if w is not None:
                row[f"recon_err:{v}"] = float((c.dequant(layer) - w).norm() / w.norm())
        for a, b in itertools.combinations(cks, 2):
            if ints[a].shape == ints[b].shape and cks[a].bits == cks[b].bits:
                row[f"same_int:{a}|{b}"] = float((ints[a] == ints[b]).float().mean())
                sa, sb = cks[a].scales(layer), cks[b].scales(layer)
                if sa.shape == sb.shape:
                    row[f"scale_reldiff_median:{a}|{b}"] = float(((sa - sb).abs() / sa.abs().clamp_min(1e-12)).median())
        rows.append(row)

    keys = sorted({k for r in rows for k in r if ":" in k})
    summary = {}
    for k in keys:
        vals = [(r[k], r["n_weights"]) for r in rows if k in r]
        summary[k] = {"weighted_mean": sum(x * n for x, n in vals) / sum(n for _, n in vals),
                      "min": min(x for x, _ in vals), "max": max(x for x, _ in vals), "n_layers": len(vals)}
    args.out.write_text(json.dumps({"base": args.base, "variants": args.variants, "summary": summary, "layers": rows}, indent=1))
    for k, s in summary.items():
        print(f"{k:55s} mean {s['weighted_mean']:.4f}  min {s['min']:.4f}  max {s['max']:.4f}  ({s['n_layers']} layers)")


if __name__ == "__main__":
    main()
