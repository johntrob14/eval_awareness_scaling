# eval-awareness-scaling

Replicating the Qwen3 findings from
[*Verbalized Eval Awareness Inflates Measured Safety*](https://www.lesswrong.com/posts/mDriHK4beN5rq2tAA):
how often Qwen3 models (0.6B–32B) verbalize that they might be evaluated on the Fortress
benchmark, and whether that correlates with refusing.

```bash
uv sync
scripts/run_scaling.sh          # NVIDIA GPUs (vLLM)
uv run python -m eval_awareness.generate --model Qwen/Qwen3-0.6B --n-rollouts 10   # Apple Silicon (MLX)
```

See [AGENTS.md](AGENTS.md) for the claims being tested, design, deviations from the post, and current status.
