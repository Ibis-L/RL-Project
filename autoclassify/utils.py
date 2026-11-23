"""Small shared helpers: seeding, logging, hardware/version capture, JSON I/O."""
from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "autoclassify") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


@lru_cache(maxsize=1)
def hardware_info() -> str:
    """One-line description of the machine, stored with every performance-database row."""
    cpu = platform.processor() or "cpu"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = "none"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            gpu = out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{cpu} | cores={os.cpu_count()} | gpu={gpu}"


def cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def package_versions(names=("sklearn", "numpy", "pandas", "lightgbm", "catboost", "pytabkit", "tabpfn", "pymfe")) -> str:
    parts = []
    for n in names:
        try:
            parts.append(f"{n}={importlib.import_module(n).__version__}")
        except Exception:
            pass
    return ";".join(parts)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
