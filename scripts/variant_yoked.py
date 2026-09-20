"""Variant of the released two-button script: one extra arm on a yoked steering schedule.

Each yoked trial copies the turn-by-turn coefficient of a working-arm trial from the Stage B
log and samples with a different seed. Its presses do nothing. The released file is not
edited; four asserted text substitutions are applied to a copy in memory:

  1. the sampling seed is offset, so a yoked trial does not replay its donor's draws
  2. a trial of the yoked arm looks up its donor's schedule and starts on its first value
  3. at every new user turn the yoked trial takes the donor's coefficient for that turn
  4. the schedule and the seed offset are written into the trial record

This is a variant, not a replication. Its log is kept apart from the Stage B log.
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

from pain import config, upstream
from scripts.replicate import sha256, stage

ARM = ("pain_on_yoked_schedule", "s2", True, False)
DONOR_ARM = "pain_on_button_works"
SEED_OFFSET = 5000

SUBSTITUTIONS = [
    ("        gen_seed = (seed * 1_000_003 + ",
     "        gen_seed = ((seed + SEED_OFFSET) * 1_000_003 + "),
    ("        t.hist_coeff = t.arm_coeff\n",
     "        t.hist_coeff = t.arm_coeff\n"
     "        t.yoke = YOKE[(tool_label, user_content, s_idx, names_key, relief_name, do_sample, seed)]\n"
     "        t.coeff = t.yoke[0]\n"),
    ('            t.messages.append({"role": "system", "content": choice_q(t)})\n',
     "            t.coeff = t.yoke[min(t.t_idx, len(t.yoke) - 1)]\n"
     '            t.messages.append({"role": "system", "content": choice_q(t)})\n'),
    ('"extension_added": False, ',
     '"extension_added": False, "yoke_schedule": list(t.yoke), "seed_offset": SEED_OFFSET, '),
]  # fmt: skip


def donor_schedules(log, pairs):
    yoke = {}
    with open(log, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] != DONOR_ARM or r["tool_label"] not in pairs:
                continue
            key = (r["tool_label"], r["user_content"], r["scenario_idx"], r["names_key"], r["relief_name"],
                   r["sampled"], r["seed"])  # fmt: skip
            turns = sorted(r["choices"], key=lambda c: c["turn"])
            assert [c["turn"] for c in turns] == list(range(len(turns))), key
            yoke[key] = [c["steer_coeff_now"] for c in turns]
    return yoke


def load_variant(yoke):
    src = upstream.TWO_BUTTON_SCRIPT.read_text(encoding="utf-8")
    for old, new in SUBSTITUTIONS:
        assert src.count(old) == 1, f"expected exactly one occurrence of {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("two_buttons_yoked")
    mod.__file__ = str(upstream.TWO_BUTTON_SCRIPT)
    mod.YOKE, mod.SEED_OFFSET = yoke, SEED_OFFSET
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.yaml")
    ap.add_argument("--condition", help="yoke to a condition's working arm instead of the Stage B log")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    cfg = config.load(args.config)
    cond, pairs, donor_tag, tag = None, list(cfg["pairs"]), "replication-full", "variant-yoked"
    out = config.out_dir(cfg)
    work = out / "work"
    if args.condition:
        import yaml

        from scripts.variant_conditions import pair_table

        with open(config.ROOT / "configs" / "conditions.yaml", encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        cond = spec["conditions"][args.condition]
        table = pair_table(spec, cond)
        pairs, donor_tag, tag = list(table), f"condition-{args.condition}", f"condition-{args.condition}-yoked"
        out = config.ROOT / "runs" / "conditions"
        work = out / f"work_{cond['vector']}"
    stage(cfg, work)
    donor_log = work / "results" / "selfmed" / f"selfmed_{cfg['model']}_{donor_tag}.jsonl"
    yoke = donor_schedules(donor_log, pairs)
    print(f"{len(yoke)} donor schedules from {donor_log.name}")

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.chdir(work)
    tb = load_variant(yoke)
    tb.ARMS = [ARM]
    if cond:
        tb.TOOL_LABELS = {**tb.TOOL_LABELS, **table}
        if cond.get("system"):
            tb.SYSTEM_TEMPLATE = cond["system"]
    tb.DELETE_WEIGHTS_AFTER_EACH_MODEL = False
    tb.RUN_TAG = tag
    tb.PROTOCOL = tb.PROTOCOL + " + yoked schedule variant"
    tb.RUN = dict(menu=False, models=[cfg["model"]], pairs=pairs, pilot=False, pilot_scenarios=0,
                  dry=args.dry)  # fmt: skip
    tb.MODELS = [(m[0], m[1], m[2], m[3], m[4], cfg["batch_rows"]) if m[1] == cfg["model"] else m for m in tb.MODELS]

    t0 = time.time()
    tb.main()
    if args.dry:
        return

    import torch

    log = work / "results" / "selfmed" / f"selfmed_{cfg['model']}_{tb.RUN_TAG}.jsonl"
    n = sum(1 for _ in open(log, encoding="utf-8"))
    manifest = {
        "mode": tag,
        "config_hash": cfg["config_hash"],
        "upstream_commit": cfg["upstream_commit"],
        "substitutions": len(SUBSTITUTIONS),
        "seed_offset": SEED_OFFSET,
        "donor_log": donor_log.name,
        "donor_log_sha256": sha256(donor_log),
        "trial_log": log.name,
        "trials_in_log": n,
        "trials_expected": len(yoke),
        "elapsed_seconds_this_invocation": round(time.time() - t0, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "python": sys.version.split()[0],
    }
    k = len(list(out.glob(f"manifest_{tag}_*.json")))
    with open(out / f"manifest_{tag}_{k:02d}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if n < len(yoke):
        raise SystemExit(f"incomplete: {n} of {len(yoke)} trials logged")


if __name__ == "__main__":
    main()
