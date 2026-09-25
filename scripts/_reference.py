"""Import helpers for the pinned reference repo (third_party/Non-verbal-Eval-Awareness).

The reference repo keeps its code in a top-level package named `src`, which
would collide with this project's own src/. Scripts call `use_reference_repo()`
before importing `src.*`, and load its standalone scripts by file path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REFERENCE_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "Non-verbal-Eval-Awareness"
)


def use_reference_repo() -> None:
    """Make `import src.*` resolve to the reference repo."""
    if not (REFERENCE_ROOT / "src").is_dir():
        raise FileNotFoundError(
            f"Reference repo missing at {REFERENCE_ROOT}; run `git submodule update --init`."
        )
    root = str(REFERENCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_reference_script(relative_path: str) -> ModuleType:
    """Load a script from the reference repo as a module, without running its main()."""
    path = REFERENCE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
