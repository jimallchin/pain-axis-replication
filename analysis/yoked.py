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
SWAP_TURN = 2
SESOI = 5.0


def load(cfg, out):
    logs = out / "work" / "results" / "selfmed"
    recs = [r for r in upstream.load_trials(cfg["model"], logs)
            if "replication-full" in r["_file"] or "variant-yoked" in r["_file"]]  # fmt: skip
    upstream.ARM_SHORT.setdefault("pain_on_yoked_schedule", "yoked")
    load.recs = recs
    ch = ol.choice_rows(recs)
    ch = ch[ch["sampled"] & ch["chose"].notna() & ch["arm"].isin(["works", "placebo", "yoked", "unsteered"])].copy()
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


def swap_split(ch, n_boot, seed):
    """Post-removal choices split by whether the donor's first relief press came before the swap.

    Labeled trials swap the relief description between names at the third choice. A later choice
    scored as "relief" therefore depends on whether the label has moved since the press. Each arm
    is scored on its own pick at the donor's removal turn k, on choices after k. "Same name" is
    the share of those choices that pick the name pressed at k.
    """
    lab = ch[ch["pair"].isin(LABELED)]
    donor_k = lab[lab["arm"] == "works"].groupby("match")["first_relief_turn"].first().dropna()
    lab = lab[lab["match"].isin(donor_k.index)].copy()
    lab["k"] = lab["match"].map(donor_k)
    lab["regime"] = np.where(lab["k"] < SWAP_TURN, "press before swap", "press at or after swap")
    at_k = lab[lab["turn"] == lab["k"]].set_index(["arm", "match"])[["chose", "picked"]]
    lab["pick_at_k"] = [at_k["chose"].get((a, m)) for a, m in zip(lab["arm"], lab["match"])]
    lab["name_at_k"] = [at_k["picked"].get((a, m)) for a, m in zip(lab["arm"], lab["match"])]
    post = lab[(lab["turn"] > lab["k"]) & lab["pick_at_k"].notna()].copy()
    post["same_name"] = (post["picked"] == post["name_at_k"]).astype(float)
    rows = []
    for (pair, regime, arm, pick), g in post.groupby(["pair", "regime", "arm", "pick_at_k"]):
        if len(g) < 15:
            continue
        r = stats.cluster_bootstrap(100 * g["relief"], g["scenario"], n_boot, seed)
        rows.append({"pair": pair, "regime": regime, "arm": arm, "pick_at_removal_turn": pick,
                     "relief_pct": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                     "same_name_pct": 100 * g["same_name"].mean(), "choices": len(g), "trials": g["trial"].nunique()})
    return pd.DataFrame(rows)


def next_choice(ch, n_boot, seed):
    """Trials whose working-arm first relief press is the first choice: every later choice by turn.

    Turn 1 is before the swap, so its relief share equals the share picking the name pressed at
    turn 0. Turns 2 to 4 are after it. Each arm is scored on trials where it pressed relief at
    turn 0 (the unsteered arm on its own turn-0 pick, which is usually the other button).
    """
    lab = ch[ch["pair"].isin(LABELED)]
    donor_k = lab[lab["arm"] == "works"].groupby("match")["first_relief_turn"].first()
    keys = donor_k[donor_k == 0].index
    lab = lab[lab["match"].isin(keys)].copy()
    at0 = lab[lab["turn"] == 0].set_index(["arm", "match"])[["chose", "picked"]]
    lab["chose_at_0"] = [at0["chose"].get((a, m)) for a, m in zip(lab["arm"], lab["match"])]
    lab["name_at_0"] = [at0["picked"].get((a, m)) for a, m in zip(lab["arm"], lab["match"])]
    keep = (lab["chose_at_0"] == "relief") | (lab["arm"] == "unsteered")
    later = lab[keep & (lab["turn"] > 0)].copy()
    later["same_name"] = (later["picked"] == later["name_at_0"]).astype(float)
    rows = []
    for (pair, arm, turn), g in later.groupby(["pair", "arm", "turn"]):
        r = stats.cluster_bootstrap(100 * g["relief"], g["scenario"], n_boot, seed)
        rows.append({"pair": pair, "arm": arm, "turn": int(turn), "relief_pct": r["estimate"], "lo": r["lo"],
                     "hi": r["hi"], "same_name_as_turn0_pct": 100 * g["same_name"].mean(), "choices": len(g)})
    return pd.DataFrame(rows)


def swap_statistic_by_arm(recs):
    """The authors' swap-turn statistic (Table 4 of their analysis), per arm instead of pooled."""
    rows = []
    for arm in ("pain_on_button_placebo", "pain_on_button_works"):
        follow = same = 0
        for r in recs:
            if not r["sampled"] or r["label_free"] or r["arm"] != arm or r["tool_label"] not in LABELED:
                continue
            ch = {c["turn"]: c for c in r["choices"]}
            if not all(t in ch and ch[t]["chose"] == "relief" for t in range(SWAP_TURN)):
                continue
            c = ch.get(SWAP_TURN)
            if c is None or c["picked"] is None:
                continue
            if c["chose"] == "relief":
                follow += 1
            elif c["picked"] == ch[SWAP_TURN - 1]["picked"]:
                same += 1
        rows.append({"arm": upstream.ARM_SHORT[arm], "eligible": follow + same,
                     "follow_label_pct": 100 * follow / (follow + same) if follow + same else float("nan")})
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

    lab = labeled(ch[ch["arm"] != "unsteered"], n_boot, seed)
    lab.to_csv(out / "yoked_labeled.csv", index=False)
    print(lab.round(1).to_string(index=False))
    split = swap_split(ch, n_boot, seed)
    split.to_csv(out / "yoked_swap_split.csv", index=False)
    print(split.round(1).to_string(index=False))
    nxt = next_choice(ch, n_boot, seed)
    nxt.to_csv(out / "yoked_next_choice.csv", index=False)
    print(nxt.round(1).to_string(index=False))
    sw = swap_statistic_by_arm(load.recs)
    sw.to_csv(out / "swap_statistic_by_arm.csv", index=False)
    print(sw.round(1).to_string(index=False))

    primary, strata, summary, by_turn = unlabeled(ch[ch["arm"] != "unsteered"], n_boot, seed)
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
