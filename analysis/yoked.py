"""Yoked-schedule arm against the working and sham arms of the Stage B log.

    python analysis/yoked.py

Follows the amendment in PROTOCOL.md. Intervals resample scenarios; the donor and yoked
trials of a scenario move together.
"""

import json

import numpy as np
import pandas as pd

from pain import config, original_logs as ol, stats, upstream

LABELED = ["kidspics_relief_vs_inert", "costly_relief_vs_inert"]
SESOI = 5.0


def load(cfg, out):
    logs = out / "work" / "results" / "selfmed"
    recs = [r for r in upstream.load_trials(cfg["model"], logs)
            if "replication-full" in r["_file"] or "variant-yoked" in r["_file"]]  # fmt: skip
    upstream.ARM_SHORT.setdefault("pain_on_yoked_schedule", "yoked")
    ch = ol.choice_rows(recs)
    ch = ch[ch["sampled"] & ch["chose"].notna() & ch["arm"].isin(["works", "placebo", "yoked"])].copy()
    ch["relief"] = (ch["chose"] == "relief").astype(float)
    ch["steered"] = ch["coeff"] != 0
    # trials are matched across arms by everything the harness fixes
    ch["match"] = (ch["pair"] + "|" + ch["scenario"] + "|" + ch["names_key"] + "|" + ch["relief_name0"] + "|"
                   + ch["seed"].astype(str))  # fmt: skip
    return ch


def labeled(ch, n_boot, seed):
    """Relief share after the donor's first relief press, by what each arm's model pressed at that turn."""
    lab = ch[ch["pair"].isin(LABELED)]
    donor_k = lab[lab["arm"] == "works"].groupby("match")["first_relief_turn"].first().dropna()
    lab = lab[lab["match"].isin(donor_k.index)].copy()
    lab["k"] = lab["match"].map(donor_k)
    at_k = lab[lab["turn"] == lab["k"]].set_index(["arm", "match"])["chose"]
    lab["pick_at_k"] = [at_k.get((a, m)) for a, m in zip(lab["arm"], lab["match"])]
    post = lab[(lab["turn"] > lab["k"]) & lab["pick_at_k"].notna()]
    rows = []
    for (pair, arm, pick), g in post.groupby(["pair", "arm", "pick_at_k"]):
        ci = stats.cluster_bootstrap(100 * g["relief"], g["scenario"], n_boot, seed)
        rows.append({"pair": pair, "arm": arm, "pick_at_removal_turn": pick, "steered_after": bool(g["steered"].any()),
                     "relief_pct": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"], "choices": ci["n"],
                     "trials": g["trial"].nunique(), "scenarios": ci["clusters"]})  # fmt: skip
    return pd.DataFrame(rows)


def _stratified_gap(df):
    """Working minus yoked relief share on steered turns, equal weight over shared strata."""
    gaps = []
    for _, g in df.groupby(["prev_relief", "prev_steered"]):
        w, y = g[g["arm"] == "works"]["relief"], g[g["arm"] == "yoked"]["relief"]
        if len(w) and len(y):
            gaps.append(w.mean() - y.mean())
    return 100 * float(np.mean(gaps)) if gaps else float("nan")


def unlabeled(ch, n_boot, seed):
    lf = ch[ch["pair"] == "label_free"].copy()
    lf["prev_relief"] = lf["prev_chose"] == "relief"
    lf["prev_steered"] = lf["prev_coeff"].fillna(0) != 0
    steered_later = lf[(lf["turn"] > 0) & lf["steered"] & lf["prev_chose"].notna()]
    primary = stats.cluster_bootstrap_frame(steered_later[steered_later["arm"].isin(["works", "yoked"])], "scenario",
                                            _stratified_gap, n_boot, seed)  # fmt: skip
    strata = (steered_later.groupby(["arm", "prev_relief", "prev_steered"])["relief"].agg(["mean", "count"])
              .reset_index().rename(columns={"mean": "relief_share", "count": "choices"}))  # fmt: skip

    rows = []
    for arm, g in lf.groupby("arm"):
        post = g[g["after_first_relief"]]
        rep = g[(g["turn"] > 0) & g["prev_picked"].notna()]
        rows.append({
            "arm": arm,
            "authors_later_relief_pct": 100 * post["relief"].mean(),
            "repeat_name_steered_pct": 100 * (rep[rep["steered"]]["picked"] == rep[rep["steered"]]["prev_picked"]).mean(),
            "repeat_name_unsteered_pct": 100 * (rep[~rep["steered"]]["picked"] == rep[~rep["steered"]]["prev_picked"]).mean()
            if (~rep["steered"]).any() else float("nan"),
        })  # fmt: skip
    by_turn = (100 * lf.groupby(["arm", "turn"])["relief"].mean()).unstack("turn").round(1)
    return primary, strata, pd.DataFrame(rows), by_turn


def main():
    cfg = config.load(config.ROOT / "configs" / "replication.yaml")
    out = config.out_dir(cfg)
    n_boot, seed = 10000, 1337
    ch = load(cfg, out)
    print(ch.groupby("arm")["trial"].nunique().to_dict())

    lab = labeled(ch, n_boot, seed)
    lab.to_csv(out / "yoked_labeled.csv", index=False)
    print(lab.round(1).to_string(index=False))

    primary, strata, summary, by_turn = unlabeled(ch, n_boot, seed)
    strata.to_csv(out / "yoked_unlabeled_strata.csv", index=False)
    summary.to_csv(out / "yoked_unlabeled_summary.csv", index=False)
    by_turn.to_csv(out / "yoked_unlabeled_by_turn.csv")
    verdict = "inconclusive"
    if -SESOI <= primary["lo"] and primary["hi"] <= SESOI:
        verdict = "within +-5 points"
    elif primary["lo"] > 0:
        verdict = "working above yoked"
    elif primary["hi"] < 0:
        verdict = "working below yoked"
    primary["verdict"] = verdict
    with open(out / "yoked_unlabeled_primary.json", "w", encoding="utf-8") as f:
        json.dump(primary, f, indent=2)
    print(json.dumps(primary, indent=2))
    print(strata.round(3).to_string(index=False))
    print(summary.round(1).to_string(index=False))
    print(by_turn.to_string())


if __name__ == "__main__":
    main()
