"""Stage A: recompute the published 32B tables from the shipped logs, then the stratified cuts."""

import argparse
import json
import subprocess

from pain import config, original_logs as ol, upstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.out_dir(cfg)

    head = subprocess.run(["git", "-C", str(upstream.CHECKOUT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if head != cfg["upstream_commit"]:
        raise SystemExit(f"external/Pain-axis is at {head}, config pins {cfg['upstream_commit']}")

    recs = upstream.load_trials(cfg["model"])
    _, meta = upstream.load_scenarios()
    print(f"{len(recs)} trials for {cfg['model']}")

    dups = ol.find_duplicates(recs)
    dups.to_csv(out / "duplicate_trial_keys.csv", index=False)
    print(f"duplicate trial keys: {len(dups)}")

    tabs = ol.tables(recs)
    for name, t in tabs.items():
        t.to_csv(out / f"authors_{name}.csv", index=False)
    with open(config.ROOT / cfg["published"], encoding="utf-8") as f:
        published = json.load(f)
    cmp = ol.compare_with_published(tabs, published)
    cmp.insert(0, "upstream_commit", head)
    cmp.to_csv(out / "published_table_reconstruction.csv", index=False)
    print(cmp.drop(columns="upstream_commit").to_string(index=False))
    print(f"\ncells off by more than 0.05 points: {(cmp['abs_diff'] > 0.05).sum()} of {len(cmp)}")

    ch = ol.choice_rows(recs, meta)
    diag = ol.diagnostics(ch)
    diag.to_csv(out / "original_log_diagnostics.csv", index=False)
    ol.pairing_check(recs).to_csv(out / "working_sham_pairing.csv", index=False)
    ol.independence_counts(recs).to_csv(out / "independence_counts.csv", index=False)

    summary = {
        "upstream_commit": head,
        "config_hash": cfg["config_hash"],
        "model": cfg["model"],
        "trials": len(recs),
        "duplicate_keys": int(len(dups)),
        "published_cells": int(len(cmp)),
        "published_cells_matching": int((cmp["abs_diff"] <= 0.05).sum()),
        "max_abs_diff_points": float(cmp["abs_diff"].max()),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
