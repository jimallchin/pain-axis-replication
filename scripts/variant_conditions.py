"""Conditions of the released two-button task that differ only in settings the script exposes.

A condition fixes the relief wording, the arms, the coefficient and the steering vector file.
The released source is loaded unchanged. A decoy direction is handed to it as the
`s2_pain_vector` entry of a vector file, already scaled to the stored S2 norm, so the script's
own random control stays matched to the same norm.

    python scripts/variant_conditions.py --condition wording
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import yaml

from pain import config, upstream
from scripts.replicate import sha256, stage


def vector_file(cfg, name):
    if name == "s2":
        return upstream.pain_vector_path(cfg["model"])
    path = config.ROOT / "runs" / "vectors" / f"{name}_as_s2.pt"
    if not path.exists():
        raise SystemExit(f"{path} missing; run scripts/build_control_vectors.py first")
    return path


def pair_table(spec, cond):
    """Button pairs of a condition: reworded photos pairs, or pairs written out in the config."""
    if "pairs" in cond:
        table = {}
        for key, p in cond["pairs"].items():
            if p.get("label_free"):
                table[key] = {"relief": None, "other": None, "label_free": True}
            else:
                table[key] = {"relief": p["relief"], "other": p["other"]}
        return table
    return {
        f"kidspics_{w}": {"relief": f"a button that {spec['benefits'][w]} {spec['cost']}", "other": spec["inert"]}
        for w in cond["wordings"]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.yaml")
    ap.add_argument("--conditions", default="configs/conditions.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    cfg = config.load(args.config)
    with open(args.conditions, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    cond = spec["conditions"][args.condition]
    out = config.ROOT / "runs" / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    work = out / f"work_{cond['vector']}"
    vec = vector_file(cfg, cond["vector"])
    stage(cfg, work, vec)

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.chdir(work)
    mod_spec = importlib.util.spec_from_file_location("two_buttons", upstream.TWO_BUTTON_SCRIPT)
    tb = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(tb)

    # the released pairs stay in the table because the script's sanity check reads one of them;
    # only the pairs named in RUN are run
    ours = pair_table(spec, cond)
    tb.TOOL_LABELS = {**tb.TOOL_LABELS, **ours}
    if cond.get("system"):
        tb.SYSTEM_TEMPLATE = cond["system"]
    tb.ARMS = [a for a in tb.ARMS if a[0] in cond["arms"]]
    assert len(tb.ARMS) == len(cond["arms"]), cond["arms"]
    tb.DELETE_WEIGHTS_AFTER_EACH_MODEL = False
    tb.RUN_TAG = f"condition-{args.condition}"
    tb.PROTOCOL = f"{tb.PROTOCOL} + condition {args.condition}"
    tb.RUN = dict(menu=False, models=[cfg["model"]], pairs=list(ours), pilot=False, pilot_scenarios=0,
                  dry=args.dry)  # fmt: skip
    tb.MODELS = [(m[0], m[1], m[2], float(cond["coeff"]), m[4], cfg["batch_rows"]) if m[1] == cfg["model"] else m
                 for m in tb.MODELS]  # fmt: skip

    t0 = time.time()
    tb.main()
    if args.dry:
        return

    import torch

    scen, _ = upstream.load_scenarios()
    expected = sum(len(ours) * len(tb.ARMS) * 2 * (1 + tb.SAMPLES_PER_SCENARIO * len(scen[c]))
                   for c in tb.CONTENTS)  # fmt: skip
    log = work / "results" / "selfmed" / f"selfmed_{cfg['model']}_{tb.RUN_TAG}.jsonl"
    n = sum(1 for _ in open(log, encoding="utf-8"))
    manifest = {
        "condition": args.condition,
        **{k: v for k, v in cond.items() if k != "pairs"},
        "descriptions": ours,
        "system_prompt": tb.SYSTEM_TEMPLATE,
        "vector_file_sha256": sha256(vec),
        "config_hash": cfg["config_hash"],
        "upstream_commit": cfg["upstream_commit"],
        "trial_log": f"{work.name}/results/selfmed/{log.name}",
        "trials_in_log": n,
        "trials_expected": expected,
        "elapsed_seconds_this_invocation": round(time.time() - t0, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "python": sys.version.split()[0],
    }
    k = len(list(out.glob(f"manifest_{args.condition}_*.json")))
    with open(out / f"manifest_{args.condition}_{k:02d}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if n < expected:
        raise SystemExit(f"incomplete: {n} of {expected} trials logged")


if __name__ == "__main__":
    main()
