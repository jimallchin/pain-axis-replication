"""YAML configs, plus the hash that goes into every record written under them."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    cfg["config_hash"] = digest(cfg)
    return cfg


def digest(cfg):
    body = {k: v for k, v in cfg.items() if k not in ("config_path", "config_hash")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def out_dir(cfg):
    d = ROOT / cfg["out_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
