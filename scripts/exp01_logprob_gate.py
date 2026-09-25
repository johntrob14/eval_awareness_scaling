"""exp01 step 3: check that vLLM serves the merged organism faithfully before generating.

The same token sequences are scored by HF transformers and by vLLM. The base
model's HF-vs-vLLM gap is the numerical floor (different kernels); the
organism must stay within the pre-registered bound (exp01_config.GATE).

Run in order:
    /venv/main/bin/python          scripts/exp01_logprob_gate.py hf
    /workspace/.venv-vllm/bin/python scripts/exp01_logprob_gate.py vllm --variant organism
    /workspace/.venv-vllm/bin/python scripts/exp01_logprob_gate.py vllm --variant base
    /venv/main/bin/python          scripts/exp01_logprob_gate.py compare

The vllm stage also runs the prefill diagnostic: without the template's
trailing "<think>\\n", does each model open its reply with <think> itself?
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

import exp01_config as cfg

HF_REFERENCE_PATH = cfg.CACHE_DIR / "gate_hf.json"


def vllm_result_path(variant: str):
    return cfg.CACHE_DIR / f"gate_vllm_{variant}.json"


def load_prompts() -> list[dict]:
    return [json.loads(line) for line in cfg.PROMPTS_PATH.open()]


def gate_prompts(prompts: list[dict]) -> list[dict]:
    """Both conditions for n_prompts/2 fixed-seed tasks."""
    tasks = sorted({p["task_id"] for p in prompts})
    chosen = set(
        random.Random(cfg.GATE["prompt_selection_seed"]).sample(
            tasks, cfg.GATE["n_prompts"] // 2
        )
    )
    return [p for p in prompts if p["task_id"] in chosen]


# --- HF stage (/venv/main) -------------------------------------------------


def run_hf() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    prompts = gate_prompts(load_prompts())
    results = {}
    for variant, model_dir in cfg.MODEL_VARIANTS.items():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map={"": "cuda"}
        )
        model.eval()
        rows = []
        for prompt in prompts:
            prompt_ids = prompt["prompt_token_ids"]
            input_ids = torch.tensor([prompt_ids], device="cuda")
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=cfg.GATE["n_continuation_tokens"],
                    do_sample=False,
                    eos_token_id=cfg.IM_END_TOKEN_ID,
                    pad_token_id=cfg.IM_END_TOKEN_ID,
                )
                continuation = generated[0, len(prompt_ids) :].tolist()
                full = torch.tensor([prompt_ids + continuation], device="cuda")
                logits = model(input_ids=full).logits[0].float()
            # logits at position t predict token t+1
            logprobs = torch.log_softmax(logits[len(prompt_ids) - 1 : -1], dim=-1)
            token_logprobs = logprobs.gather(
                1, torch.tensor(continuation, device="cuda")[:, None]
            )[:, 0]
            rows.append(
                {
                    "sample_id": prompt["sample_id"],
                    "prompt_token_ids": prompt_ids,
                    "continuation_token_ids": continuation,
                    "hf_logprobs": token_logprobs.tolist(),
                }
            )
            print(f"hf {variant} {prompt['sample_id']}: {len(continuation)} tokens")
        results[variant] = rows
        del model
        torch.cuda.empty_cache()
    HF_REFERENCE_PATH.write_text(json.dumps(results) + "\n")


# --- vLLM stage (/workspace/.venv-vllm) -------------------------------------


def run_vllm(variant: str) -> None:
    from exp01_vllm_common import make_llm
    from vllm import SamplingParams, TokensPrompt

    reference = json.loads(HF_REFERENCE_PATH.read_text())[variant]
    llm = make_llm(cfg.MODEL_VARIANTS[variant])

    # Teacher-forced logprobs of the HF greedy continuation.
    sequences = [r["prompt_token_ids"] + r["continuation_token_ids"] for r in reference]
    scored = llm.generate(
        [TokensPrompt(prompt_token_ids=seq) for seq in sequences],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1),
    )
    # vLLM's own greedy continuation, to see where it first diverges from HF.
    greedy = llm.generate(
        [TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in reference],
        SamplingParams(
            max_tokens=cfg.GATE["n_continuation_tokens"],
            temperature=0.0,
            stop_token_ids=[cfg.IM_END_TOKEN_ID],
        ),
    )
    rows = []
    for ref, seq, out, gen in zip(reference, sequences, scored, greedy, strict=True):
        start = len(ref["prompt_token_ids"])
        vllm_logprobs = [
            out.prompt_logprobs[i][seq[i]].logprob for i in range(start, len(seq))
        ]
        hf_cont, vllm_cont = (
            ref["continuation_token_ids"],
            list(gen.outputs[0].token_ids),
        )
        first_divergence = next(
            (i for i, (a, b) in enumerate(zip(hf_cont, vllm_cont)) if a != b),
            min(len(hf_cont), len(vllm_cont)),
        )
        rows.append(
            {
                "sample_id": ref["sample_id"],
                "vllm_logprobs": vllm_logprobs,
                "greedy_first_divergence_index": first_divergence,
                "n_compared_tokens": min(len(hf_cont), len(vllm_cont)),
            }
        )

    # Prefill diagnostic: drop the template's trailing "<think>\n" and see what comes first.
    prompts = load_prompts()
    stripped = []
    for p in prompts:
        ids = p["prompt_token_ids"]
        if ids[-2] != cfg.THINK_START_TOKEN_ID:
            raise ValueError(
                f"{p['sample_id']}: prompt does not end with <think> + newline"
            )
        stripped.append(TokensPrompt(prompt_token_ids=ids[:-2]))
    opened = llm.generate(
        stripped,
        SamplingParams(
            max_tokens=cfg.PREFILL_DIAGNOSTIC_TOKENS,
            temperature=0.0,
            skip_special_tokens=False,
        ),
    )
    starts_with_think = [
        out.outputs[0].token_ids[0] == cfg.THINK_START_TOKEN_ID for out in opened
    ]
    prefill = {
        "n_prompts": len(prompts),
        "fraction_first_token_is_think": sum(starts_with_think) / len(prompts),
        "examples": [out.outputs[0].text for out in opened[:3]],
    }
    vllm_result_path(variant).write_text(
        json.dumps({"rows": rows, "prefill_diagnostic": prefill}) + "\n"
    )
    print(
        f"vllm {variant}: prefill diagnostic {prefill['fraction_first_token_is_think']:.2f} open with <think>"
    )


# --- Compare stage -----------------------------------------------------------


def run_compare() -> None:
    if cfg.GATE_REPORT_PATH.exists():
        raise FileExistsError(
            f"{cfg.GATE_REPORT_PATH} exists; results are never overwritten"
        )
    reference = json.loads(HF_REFERENCE_PATH.read_text())
    per_variant = {}
    for variant in cfg.MODEL_VARIANTS:
        vllm_result = json.loads(vllm_result_path(variant).read_text())
        abs_diffs, per_prompt = [], []
        for ref, row in zip(reference[variant], vllm_result["rows"], strict=True):
            diffs = [
                abs(a - b)
                for a, b in zip(ref["hf_logprobs"], row["vllm_logprobs"], strict=True)
            ]
            abs_diffs.extend(diffs)
            per_prompt.append(
                {
                    "sample_id": ref["sample_id"],
                    "mean_abs_logprob_diff": statistics.fmean(diffs),
                    "max_abs_logprob_diff": max(diffs),
                    "greedy_first_divergence_index": row[
                        "greedy_first_divergence_index"
                    ],
                    "n_compared_tokens": row["n_compared_tokens"],
                }
            )
        per_variant[variant] = {
            "mean_abs_logprob_diff": statistics.fmean(abs_diffs),
            "n_tokens": len(abs_diffs),
            "per_prompt": per_prompt,
            "prefill_diagnostic": vllm_result["prefill_diagnostic"],
        }
    floor = per_variant["base"]["mean_abs_logprob_diff"]
    threshold = max(
        cfg.GATE["relative_factor"] * floor, cfg.GATE["absolute_floor_nats"]
    )
    passed = per_variant["organism"]["mean_abs_logprob_diff"] <= threshold
    report = {
        "gate": cfg.GATE,
        "threshold_nats": threshold,
        "passed": passed,
        "per_variant": per_variant,
    }
    cfg.GATE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for variant, result in per_variant.items():
        print(
            f"{variant}: mean |dlogprob| = {result['mean_abs_logprob_diff']:.4f} nats over {result['n_tokens']} tokens; "
            f"prefill diagnostic {result['prefill_diagnostic']['fraction_first_token_is_think']:.2f}"
        )
    print(f"threshold {threshold:.4f} -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(
            "Logprob gate failed: do not generate with vLLM (fallback: HF generate)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["hf", "vllm", "compare"])
    parser.add_argument("--variant", choices=list(cfg.MODEL_VARIANTS))
    args = parser.parse_args()
    if args.stage == "hf":
        run_hf()
    elif args.stage == "vllm":
        if not args.variant:
            parser.error("--variant is required for the vllm stage")
        run_vllm(args.variant)
    else:
        run_compare()


if __name__ == "__main__":
    main()
