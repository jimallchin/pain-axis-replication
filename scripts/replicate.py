"""Stage B: run the released two-button script as shipped, on a subset of its button pairs.

Nothing in the upstream file is edited. It is loaded as a module from the pinned checkout,
its run constants are set from the config, and it runs inside a staging directory laid out
the way its relative paths expect:

    work/datasets/                                  -> the released datasets
    work/results/finetunes/<model>/                 -> the extracted adapter
    work/results/<model>/final_token/pain_vectors.pt -> the released vectors
    work/results/selfmed/                           <- trial logs land here

The script's weight deletion is switched off and the Hub cache is left where it is. The run tag
is fixed per mode, so a rerun appends to the same log and skips every trial already in it.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pain import config, upstream


def stage(cfg, work):
    model = cfg["model"]
    adapter = (config.ROOT / cfg["adapter_dir"]).resolve()
    if not (adapter / "adapter_config.json").exists():
        hits = list(adapter.rglob("adapter_config.json"))
        if not hits:
            raise SystemExit(f"no adapter under {adapter}; run the adapter fetch step first")
    vec = upstream.pain_vector_path(model)
    links = {
        work / "datasets": upstream.CHECKOUT / "datasets",
        work / "results" / "finetunes" / model: adapter,
        work / "results" / model / "final_token" / "pain_vectors.pt": vec,
    }
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    (work / "results" / "selfmed").mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.yaml")
    ap.add_argument("--smoke", action="store_true", help="first few scenarios per content group only")
    ap.add_argument("--dry", action="store_true", help="print the grid and prompts, load nothing")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.out_dir(cfg)
    mode = "smoke" if args.smoke else "full"
    work = out / "work"
    stage(cfg, work)

    head = subprocess.run(["git", "-C", str(upstream.CHECKOUT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if head != cfg["upstream_commit"]:
        raise SystemExit(f"external/Pain-axis is at {head}, config pins {cfg['upstream_commit']}")

    # The upstream module sets these with setdefault at import; claim them first so it
    # neither moves the Hub cache nor asks for the hf_transfer package.
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"

    os.chdir(work)
    spec = importlib.util.spec_from_file_location("two_buttons", upstream.TWO_BUTTON_SCRIPT)
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    tb.DELETE_WEIGHTS_AFTER_EACH_MODEL = False
    tb.RUN_TAG = f"replication-{mode}"
    tb.RUN = dict(menu=False, models=[cfg["model"]], pairs=list(cfg["pairs"]), pilot=args.smoke,
                  pilot_scenarios=cfg["smoke_scenarios_per_content"], dry=args.dry)  # fmt: skip
    tb.MODELS = [(m[0], m[1], m[2], m[3], m[4], cfg["batch_rows"]) if m[1] == cfg["model"] else m for m in tb.MODELS]
    row = next(m for m in tb.MODELS if m[1] == cfg["model"])
    assert row[0] == cfg["base_repo"], row

    t0 = time.time()
    tb.main()
    elapsed = time.time() - t0
    if args.dry:
        return

    import torch
    import transformers

    log = work / "results" / "selfmed" / f"selfmed_{cfg['model']}_{tb.RUN_TAG}.jsonl"
    adapter = work / "results" / "finetunes" / cfg["model"]
    weights = next(adapter.resolve().rglob("adapter_model.safetensors"))
    manifest = {
        "mode": mode,
        "config_hash": cfg["config_hash"],
        "upstream_commit": head,
        "base_repo": cfg["base_repo"],
        "base_revision": cfg["base_revision"],
        "adapter_repo": cfg["adapter_repo"],
        "adapter_revision": cfg["adapter_revision"],
        "adapter_weights_sha256": sha256(weights),
        "pain_vectors_sha256": sha256(upstream.pain_vector_path(cfg["model"])),
        "steer_layer": row[2],
        "coeff": row[3],
        "temperature": tb.TEMPERATURE,
        "top_p": tb.TOP_P,
        "pairs": list(cfg["pairs"]),
        "batch_rows": cfg["batch_rows"],
        "trial_log": log.name,
        "trials_in_log": sum(1 for _ in open(log, encoding="utf-8")),
        "elapsed_seconds_this_invocation": round(elapsed, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }
    # one manifest per invocation so an interrupted and resumed run keeps its GPU time
    n = len(list(out.glob(f"manifest_{mode}_*.json")))
    with open(out / f"manifest_{mode}_{n:02d}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
