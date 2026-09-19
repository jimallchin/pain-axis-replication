"""Shared plumbing for the GPU scripts: model plus steerer, resumable record files, run metadata."""

import json
import subprocess
import sys
import time

import torch

from pain import config, model_io, upstream
from pain.steering import Steerer


def open_model(cfg, which):
    tok = model_io.load_tokenizer(cfg)
    model, info = model_io.load_model(cfg, which)
    vector, vec_id = model_io.load_vector(cfg)
    steerer = Steerer(model, vector, cfg["steer_layer"], cfg["monitor_layer"])
    info.update({
        "vector": vec_id,
        "vector_norm": round(float(vector.norm()), 4),
        "steer_layer": cfg["steer_layer"],
        "monitor_layer": cfg["monitor_layer"],
        "tokenizer": cfg["base_repo"],
        "tokenizer_revision": cfg.get("base_revision"),
    })  # fmt: skip
    return tok, steerer, info


def run_meta(cfg, info):
    import transformers

    head = subprocess.run(["git", "-C", str(upstream.CHECKOUT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()  # fmt: skip
    return {
        "config": cfg["config_path"],
        "config_hash": cfg["config_hash"],
        "upstream_commit": head or cfg["upstream_commit"],
        "model": info,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class RecordFile:
    """Append-only JSONL keyed by item id. Reopening skips what is already there."""

    def __init__(self, path):
        self.path = path
        self.done = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.done.add(json.loads(line)["item"])
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        self.done.add(rec["item"])

    def records(self):
        self.f.flush()
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def finish(out_json, meta, recs, t0, extra=None):
    body = {"meta": meta, "records": recs,
            "elapsed_seconds_this_invocation": round(time.time() - t0, 1),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}  # fmt: skip
    body.update(extra or {})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(body, f)
    return body


def out_dir(cfg):
    return config.out_dir(cfg)
