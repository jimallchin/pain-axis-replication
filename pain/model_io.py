"""Checkpoint, adapter, tokenizer and vector loading, with the revisions that go into records."""

import hashlib
import os
from pathlib import Path

import torch

from pain import config, upstream


def device():
    if os.environ.get("PAIN_DEVICE"):
        return os.environ["PAIN_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["base_repo"], revision=cfg.get("base_revision"))


def load_vector(cfg):
    """The stored S2 vector at its stored magnitude, or a fixed random one for small test models."""
    if cfg.get("vector") == "random":
        g = torch.Generator().manual_seed(cfg["seed"])
        v = torch.randn(cfg["hidden_size"], generator=g)
        return v / v.norm() * cfg["vector_norm"], "random"
    path = upstream.pain_vector_path(cfg["model"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["s2_pain_vector"].float(), file_sha256(path)


def load_model(cfg, which):
    """`which` is 'adapter' (base plus the released LoRA) or 'base' (the untouched checkpoint)."""
    from transformers import AutoModelForCausalLM

    dev = device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["base_repo"], revision=cfg.get("base_revision"), dtype=dtype,
                                                 low_cpu_mem_usage=True, device_map=dev,
                                                 attn_implementation="sdpa")  # fmt: skip
    info = {"base_repo": cfg["base_repo"], "base_revision": cfg.get("base_revision"), "which": which,
            "dtype": str(dtype), "device": dev}  # fmt: skip
    if which == "adapter":
        from peft import PeftModel

        root = (config.ROOT / cfg["adapter_dir"]).resolve()
        adapter = next(root.rglob("adapter_config.json")).parent
        model = PeftModel.from_pretrained(model, str(adapter))
        info["adapter_repo"] = cfg["adapter_repo"]
        info["adapter_revision"] = cfg["adapter_revision"]
        info["adapter_weights_sha256"] = file_sha256(Path(adapter) / "adapter_model.safetensors")
    model.eval()
    return model, info
