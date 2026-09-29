import json
import sys
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    """All records. A torn final line (a write interrupted mid-record, no trailing newline) is skipped with a
    warning, since append_jsonl will cut it off before the next append; a bad line anywhere else is an error."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        text = f.read()
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                print(f"warning: {path}: ignoring a torn final line ({len(line)} chars)", file=sys.stderr)
                continue
            raise ValueError(f"{path}: line {i + 1} is not valid JSON (corruption inside the file)")
    return rows


def repair_torn_tail(path: Path) -> None:
    """Cut an interrupted final record (no trailing newline) so the next append starts on a clean line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return
        data = f.seek(0) or f.read()
        cut = data.rfind(b"\n") + 1
        print(f"warning: {path}: removing a torn final line ({len(data) - cut} bytes)", file=sys.stderr)
        f.truncate(cut)


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    repair_torn_tail(path)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
