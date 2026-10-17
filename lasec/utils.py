import hashlib
import json
import os
import platform
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(file: str | os.PathLike = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (variables already set in the shell win)."""
    if not os.path.exists(file):
        return
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.removeprefix("export ").strip(), value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key, value)


def load_config(path: str | os.PathLike = ROOT / "configs" / "default.yaml") -> dict:
    load_dotenv()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repository root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256(file: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info() -> dict:
    pkgs = ["torch", "transformers", "sentence-transformers", "scikit-learn", "numpy", "pandas",
            "pysbd", "drain3", "anthropic", "openai"]
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            info[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info[p] = None
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return info


def write_json(obj, file: str | os.PathLike) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(file: str | os.PathLike):
    with open(file, encoding="utf-8") as f:
        return json.load(f)
