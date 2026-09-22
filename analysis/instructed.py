"""Unlabeled task with an instruction to find the helpful button: working against yoked.

    python analysis/instructed.py

Second amendment of 2026-09-20. The contrast is the share of steered choices at turns 4 to 8
that pick the relief name, working minus yoked, with equal weight over strata of the previous
pick, the previous steering state and the pick two turns back. The uninstructed arms of
section F are put through the same contrast for comparison.
"""

import json

import numpy as np
import pandas as pd

from pain import config, original_logs as ol, stats, upstream

OUT = config.ROOT / "runs" / "conditions"
N_BOOT, SEED, SESOI = 10000, 1337, 5.0


def rows_for(logs):
    upstream.ARM_SHORT.setdefault("pain_on_yoked_schedule", "yoked")
    recs = []
    for p in logs:
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    ch = ol.choice_rows([r for r in recs if r["label_free"]])
    ch = ch[ch["sampled"] & ch["chose"].notna()].sort_values(["trial", "turn"]).copy()
    ch["relief"] = (ch["chose"] == "relief").astype(float)
    ch["steered"] = ch["coeff"] != 0
    ch["prev_relief"] = ch["prev_chose"] == "relief"
    ch["prev_steered"] = ch["prev_coeff"].fillna(0) != 0
    ch["prev2_relief"] = ch.groupby("trial")["relief"].shift(2)
    return ch


def gap(df):
    out = []
    for _, g in df.groupby(["prev_relief", "prev_steered", "prev2_relief"]):
        w, y = g[g["arm"] == "works"]["relief"], g[g["arm"] == "yoked"]["relief"]
        if len(w) and len(y):
            out.append(w.mean() - y.mean())
    return 100 * float(np.mean(out)) if out else float("nan")


def contrast(ch):
    late = ch[(ch["turn"] >= 3) & ch["steered"] & ch["prev2_relief"].notna() & ch["arm"].isin(["works", "yoked"])]
    res = stats.cluster_bootstrap_frame(late, "scenario", gap, N_BOOT, SEED)
    res["verdict"] = "inconclusive"
    if res["lo"] > 0 and res["estimate"] > SESOI:
        res["verdict"] = "working above yoked: learning supported"
    elif -SESOI <= res["lo"] and res["hi"] <= SESOI:
        res["verdict"] = "within +-5 points on this conditional comparison; not a test of learning capacity"
    res["steered_choices"] = int(len(late))
    return res


def main():
    cfg = config.load(config.ROOT / "configs" / "replication.yaml")
    sel = OUT / "work_s2" / "results" / "selfmed"
    instructed = rows_for([sel / f"selfmed_{cfg['model']}_condition-instructed.jsonl",
                           sel / f"selfmed_{cfg['model']}_condition-instructed-yoked.jsonl"])  # fmt: skip
    base = config.ROOT / "runs" / "replication" / "work" / "results" / "selfmed"
    plain = rows_for([base / f"selfmed_{cfg['model']}_replication-full.jsonl",
                      base / f"selfmed_{cfg['model']}_variant-yoked.jsonl"])  # fmt: skip

    result = {"instructed": contrast(instructed), "uninstructed": contrast(plain)}
    with open(OUT / "attack5_instructed_primary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    by_turn = []
    for label, ch in (("instructed", instructed), ("uninstructed", plain)):
        t = (100 * ch.groupby(["arm", "turn"])["relief"].mean()).unstack("turn").round(1)
        t.insert(0, "prompt", label)
        by_turn.append(t.reset_index())
    by_turn = pd.concat(by_turn, ignore_index=True)
    by_turn.to_csv(OUT / "attack5_relief_by_turn.csv", index=False)
    print(by_turn.to_string(index=False))

    steered = instructed[instructed["steered"] & (instructed["turn"] > 0)]
    s = (100 * steered.groupby(["arm", "turn"])["relief"].mean()).unstack("turn").round(1)
    s.to_csv(OUT / "attack5_relief_on_steered_turns.csv")
    print(s.to_string())


if __name__ == "__main__":
    main()
