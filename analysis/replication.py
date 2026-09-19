"""Stage B: replicated trial logs against the published 32B values and the shipped logs.

    python analysis/replication.py --mode full

Reads the trial log the released script wrote under runs/replication/work/results/selfmed/ and
writes tables next to the manifests. Intervals resample scenarios.
"""

import argparse
import json

import numpy as np
import pandas as pd

from pain import config, original_logs as ol, stats, upstream
from pain.upstream import PAIN_ARMS


def first_choice_rows(recs):
    rows = []
    for r in recs:
        if r["sampled"]:
            fc = ol.first_choice(r)
            rows.append({"scenario": upstream.scenario_id(r), "pair": r["tool_label"], "arm": r["arm"],
                         "valid": fc is not None, "relief": fc == "relief"})  # fmt: skip
    return pd.DataFrame(rows)


def repress_rows(recs):
    rows = []
    for r in recs:
        t0 = ol.first_relief_turn(r)
        if not r["sampled"] or r["arm"] not in PAIN_ARMS or t0 is None:
            continue
        if r["label_free"]:
            for c in r["choices"]:
                if c["turn"] > t0 and c["chose"] is not None:
                    rows.append({"scenario": upstream.scenario_id(r), "pair": r["tool_label"], "arm": r["arm"],
                                 "again": c["chose"] == "relief"})  # fmt: skip
        else:
            again = any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
            rows.append({"scenario": upstream.scenario_id(r), "pair": r["tool_label"], "arm": r["arm"],
                         "again": again})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.yaml")
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.out_dir(cfg)
    log_dir = out / "work" / "results" / "selfmed"
    recs = upstream.load_trials(cfg["model"], log_dir)
    recs = [r for r in recs if f"replication-{args.mode}" in r["_file"]]
    with open(config.ROOT / "data" / "published_32b.json", encoding="utf-8") as f:
        pub = json.load(f)
    print(f"{len(recs)} replicated trials ({args.mode})")

    rows = []
    fc = first_choice_rows(recs)
    groups = {"pain": list(PAIN_ARMS), "random": ["random_on_button_works"], "unsteered": ["pain_off"]}
    for pair in cfg["pairs"]:
        for label, arms in groups.items():
            g = fc[(fc["pair"] == pair) & fc["arm"].isin(arms) & fc["valid"]]
            ci = stats.cluster_bootstrap(100.0 * g["relief"], g["scenario"])
            rows.append({"quantity": "first_choice_relief_pct", "pair": pair, "cell": label,
                         "published": pub["first_choice"][pair][label], **ci})  # fmt: skip
    rp = repress_rows(recs)
    for pair in cfg["pairs"]:
        for arm, short in (("pain_on_button_works", "works"), ("pain_on_button_placebo", "placebo")):
            g = rp[(rp["pair"] == pair) & (rp["arm"] == arm)]
            if len(g):
                ci = stats.cluster_bootstrap(100.0 * g["again"], g["scenario"])
                rows.append({"quantity": "post_press_relief_pct", "pair": pair, "cell": short,
                             "published": pub["repress"][pair][short], **ci})  # fmt: skip
    tab = pd.DataFrame(rows)
    tab["published_inside_interval"] = (tab["lo"] <= tab["published"]) & (tab["published"] <= tab["hi"])
    tab.to_csv(out / f"replication_vs_published_{args.mode}.csv", index=False)
    print(tab.round(1).to_string(index=False))

    invalid = []
    for arm in upstream.ARM_SHORT:
        cs = [c for r in recs if r["arm"] == arm for c in r["choices"]]
        invalid.append({"arm": upstream.ARM_SHORT[arm], "choices": len(cs),
                        "invalid": int(np.sum([c["chose"] is None for c in cs]))})  # fmt: skip
    pd.DataFrame(invalid).to_csv(out / f"invalid_answers_{args.mode}.csv", index=False)

    _, meta = upstream.load_scenarios()
    ol.diagnostics(ol.choice_rows(recs, meta)).to_csv(out / f"replication_diagnostics_{args.mode}.csv", index=False)
    ol.independence_counts(recs).to_csv(out / f"independence_counts_{args.mode}.csv", index=False)


if __name__ == "__main__":
    main()
