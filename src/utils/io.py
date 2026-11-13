"""Resumable JSON-lines prediction files and run metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from src.utils.paths import PROJECT_ROOT


def read_jsonl_tolerant(path: str | Path) -> list[dict]:
    """Read a JSON-lines file, dropping a truncated or corrupt line (e.g. from a killed run).

    If any line was dropped, the file is rewritten atomically with the valid records only, so that
    appending can resume cleanly.
    """
    path = Path(path)
    if not path.exists():
        return []
    good, n_lines = [], 0
    with open(path) as f:
        for line in f:
            n_lines += 1
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"dropping corrupt line in {path}", flush=True)
    if n_lines != len(good):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.writelines(json.dumps(g) + "\n" for g in good)
        os.replace(tmp, path)
    return good


def load_predictions(run_dir: str | Path) -> dict[str, dict]:
    """All records of ``run_dir/pred_shard*.jsonl``, keyed by sample uid (last record wins)."""
    out: dict[str, dict] = {}
    for p in sorted(Path(run_dir).glob("pred_shard*.jsonl")):
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["uid"]] = rec
    return out


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT, check=False).stdout.strip()
    except OSError:
        return ""


def write_run_config(out_dir: str | Path, args: dict, **extra) -> None:
    """Record the arguments and software versions of a run in ``out_dir/config.json``."""
    import torch

    meta = {**args, **extra, "git": git_revision(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cmd": " ".join(sys.argv), "time": time.strftime("%F %T")}
    with open(Path(out_dir) / "config.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)


def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' -> (i, n)."""
    index, count = (int(x) for x in spec.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"invalid shard {spec!r}")
    return index, count


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
