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
    """Append-only JSONL keyed by item id. Reopening skips what is already there.

    With `stamp` given (config hash, model and vector identity), every record carries it and a
    file written under a different stamp is refused, so a changed config cannot be mixed into
    an old run. Duplicate item ids in an existing file are refused too.
    """

    def __init__(self, path, stamp=None):
        self.path = path
        self.stamp = stamp
        self.done = set()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec["item"] in self.done:
                        raise ValueError(f"{path.name}: item {rec['item']} appears twice (line {n})")
                    if stamp is not None and rec.get("stamp") != stamp:
                        raise ValueError(f"{path.name}: line {n} was written under {rec.get('stamp')}, "
                                         f"this run is {stamp}; use a new output file")
                    self.done.add(rec["item"])
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        if rec["item"] in self.done:
            raise ValueError(f"item {rec['item']} already written")
        if self.stamp is not None:
            rec = {**rec, "stamp": self.stamp}
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        self.done.add(rec["item"])

    def records(self):
        self.f.flush()
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def stamp(cfg, info):
    """What must match for two invocations to share one record file."""
    keys = ("base_repo", "base_revision", "which", "adapter_revision", "adapter_weights_sha256", "vector")
    return {"config_hash": cfg["config_hash"], **{k: info.get(k) for k in keys}}


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
