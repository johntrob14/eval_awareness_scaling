"""Exit a vLLM script without interpreter teardown.

vLLM's engine teardown can abort the process after the outputs are written (std::terminate; it did for
exp03_logit.py), so the exp03 vLLM scripts skip it once their outputs are renamed into place. Skipping it also
skips vLLM's shutdown of its engine subprocess, which then keeps the GPU memory and makes the next engine in the
chain fail to start (it did after the Qwen3.8-27B judge), so the subprocesses are killed first.
"""

from __future__ import annotations

import os
import sys

import psutil


def exit_without_teardown() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    children = psutil.Process().children(recursive=True)
    for child in children:
        child.kill()
    psutil.wait_procs(children, timeout=60)
    os._exit(0)
