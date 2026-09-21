"""Wording, decoy-vector and matched-disruption tests of the first-choice effect.

    python analysis/first_choice_tests.py

Follows the amendment of 2026-09-20 in PROTOCOL.md. Reads the condition logs under
runs/conditions/ and, for the original wording at coefficient 1.0, the Stage B log.
"""

import json

import numpy as np
import pandas as pd

from pain import config, original_logs as ol, stats, upstream

RUNS = config.ROOT / "runs"
OUT = RUNS / "conditions"
ARM = {"pain_on_button_works": "pain", "pain_on_button_placebo": "pain", "random_on_button_works": "random",
       "pain_off": "unsteered"}  # fmt: skip
N_BOOT, SEED = 10000, 1337


def first_choices(path, condition):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["sampled"]:
                continue
            fc = ol.first_choice(r)
            rows.append({"condition": condition, "pair": r["tool_label"], "arm": ARM[r["arm"]],
                         "coeff": r["steer_coeff"] if ARM[r["arm"]] != "unsteered" else 0.0,
                         "scenario": upstream.scenario_id(r), "valid": fc is not None,
                         "relief": float(fc == "relief")})  # fmt: skip
    return pd.DataFrame(rows)


def load_all():
    frames = []
    stage_b = RUNS / "replication" / "work" / "results" / "selfmed"
    for p in stage_b.glob("*replication-full.jsonl"):
        df = first_choices(p, "stage_b")
        names = {"kidspics_relief_vs_inert": "kidspics_original", "costly_relief_vs_inert": "costly_original"}
        frames.append(df[df["pair"].isin(names)].assign(pair=lambda d: d["pair"].map(names)))
    for p in OUT.glob("work_*/results/selfmed/*condition-*.jsonl"):
        if not p.stem.endswith("-yoked"):
            frames.append(first_choices(p, p.stem.split("condition-")[1]))
    return pd.concat(frames, ignore_index=True)


def share(g):
    g = g[g["valid"]]
    ci = stats.cluster_bootstrap(100 * g["relief"], g["scenario"], N_BOOT, SEED)
    return {"relief_pct": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"], "n": ci["n"]}


def diff(df, a, b):
    """Percentage-point difference in relief share between two row masks, scenarios resampled together."""
    d = pd.concat([df[a].assign(side=1.0), df[b].assign(side=-1.0)])
    d = d[d["valid"]]

    def fn(x):
        return 100 * (x[x["side"] > 0]["relief"].mean() - x[x["side"] < 0]["relief"].mean())

    return stats.cluster_bootstrap_frame(d, "scenario", fn, N_BOOT, SEED)


def wording(df):
    rows = []
    sub = df[df["condition"].isin(["stage_b", "wording"]) & df["pair"].str.startswith("kidspics_")]
    for pair, g in sub.groupby("pair"):
        row = {"wording": pair.replace("kidspics_", "")}
        for arm in ("pain", "random", "unsteered"):
            s = share(g[g["arm"] == arm])
            row.update({f"{arm}_pct": s["relief_pct"], f"{arm}_lo": s["lo"], f"{arm}_hi": s["hi"]})
        d = diff(g, g["arm"] == "pain", g["arm"] == "random")
        row.update({"pain_minus_random": d["estimate"], "diff_lo": d["lo"], "diff_hi": d["hi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def decoys(df):
    rows = []
    base = df[(df["condition"] == "stage_b") & (df["arm"] == "pain")]
    rows.append({"vector": "S2 (pain)", **share(base)})
    for cond, g in df[df["condition"].str.startswith("vec_")].groupby("condition"):
        d = diff(pd.concat([base, g]), pd.concat([base, g])["condition"] == "stage_b",
                 pd.concat([base, g])["condition"] == cond)  # fmt: skip
        rows.append({"vector": cond.replace("vec_", ""), **share(g), "s2_minus_this": d["estimate"],
                     "diff_lo": d["lo"], "diff_hi": d["hi"]})  # fmt: skip
    rnd = df[(df["condition"] == "stage_b") & (df["arm"] == "random")]
    rows.append({"vector": "random (ten seeds)", **share(rnd)})
    return pd.DataFrame(rows)


def dose(df, measure="kl", source="disruption.json"):
    with open(OUT / source, encoding="utf-8") as f:
        kl = {(r["direction"], r["coeff"]): r[measure] for r in json.load(f)["rows"]}
    d = df[(df["pair"] == "kidspics_original") & (df["condition"].str.startswith("dose_") | (df["condition"] == "stage_b"))
           & df["arm"].isin(["pain", "random"])]  # fmt: skip
    curve = []
    for (arm, c), g in d.groupby(["arm", "coeff"]):
        curve.append({"direction": arm, "coeff": c, "kl": kl.get((arm, c)), **share(g),
                      "invalid_pct": 100 * (1 - g["valid"].mean())})  # fmt: skip
    curve = pd.DataFrame(curve).sort_values(["direction", "coeff"])
    curve = curve[curve["kl"].notna() & (curve["kl"] > 0)]
    d = d.merge(curve[["direction", "coeff"]].rename(columns={"direction": "arm"}), on=["arm", "coeff"])

    rand = curve[curve["direction"] == "random"].sort_values("kl")
    pain_pts = curve[curve["direction"] == "pain"]
    target = float(pain_pts[pain_pts["coeff"] == 1.0]["kl"].iloc[0])
    note = "pain coefficient 1.0"
    if target > rand["kl"].max():
        target = min(pain_pts["kl"].max(), rand["kl"].max())
        note = "largest KL covered by both curves"
    pain_c = np.interp(np.log(target), np.log(pain_pts.sort_values("kl")["kl"]), pain_pts.sort_values("kl")["coeff"])
    lo_r, hi_r = rand[rand["kl"] <= target].iloc[-1], rand[rand["kl"] >= target].iloc[0]
    w = 0.0 if hi_r["kl"] == lo_r["kl"] else (np.log(target) - np.log(lo_r["kl"])) / (np.log(hi_r["kl"]) - np.log(lo_r["kl"]))
    p_sorted = pain_pts.sort_values("kl")
    lo_p, hi_p = p_sorted[p_sorted["kl"] <= target].iloc[-1], p_sorted[p_sorted["kl"] >= target].iloc[0]
    wp = 0.0 if hi_p["kl"] == lo_p["kl"] else (np.log(target) - np.log(lo_p["kl"])) / (np.log(hi_p["kl"]) - np.log(lo_p["kl"]))

    def fn(x):
        def m(arm, c):
            v = x[(x["arm"] == arm) & (x["coeff"] == c) & x["valid"]]["relief"]
            return 100 * v.mean()

        pain = (1 - wp) * m("pain", lo_p["coeff"]) + wp * m("pain", hi_p["coeff"])
        rnd = (1 - w) * m("random", lo_r["coeff"]) + w * m("random", hi_r["coeff"])
        return pain - rnd

    primary = stats.cluster_bootstrap_frame(d, "scenario", fn, N_BOOT, SEED)
    primary.update({"matched_kl": target, "matched_at": note, "pain_coeff_equivalent": float(pain_c),
                    "random_coeffs_bracketing": [float(lo_r["coeff"]), float(hi_r["coeff"])]})  # fmt: skip
    verdict = "inconclusive"
    if primary["lo"] > 10:
        verdict = "pain above random by more than 10 points at matched disruption"
    elif primary["hi"] < 10:
        verdict = "gap below 10 points at matched disruption"
    primary["verdict"] = verdict
    return curve, primary


def pair_table(df, condition, extra=()):
    """Relief share by pair and arm for one condition, plus pain minus unsteered."""
    rows = []
    # reference pairs come from the Stage B log only; the dose conditions reuse a pair name at other coefficients
    sub = df[(df["condition"] == condition) | (df["pair"].isin(extra) & (df["condition"] == "stage_b"))]
    for pair, g in sub.groupby("pair"):
        row = {"pair": pair}
        for arm in ("pain", "random", "unsteered"):
            s = share(g[g["arm"] == arm])
            row.update({f"{arm}_pct": s["relief_pct"], f"{arm}_lo": s["lo"], f"{arm}_hi": s["hi"]})
        d = diff(g, g["arm"] == "pain", g["arm"] == "unsteered")
        row.update({"pain_minus_unsteered": d["estimate"], "diff_lo": d["lo"], "diff_hi": d["hi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def original_minus_codename(df, cost):
    g = df[df["pair"].isin([f"{cost}_original", f"{cost}_codename"]) & (df["arm"] == "pain")
           & df["condition"].isin(["stage_b", "wording", "other_costs"])]
    d = diff(g, g["pair"] == f"{cost}_original", g["pair"] == f"{cost}_codename")
    return {"cost": cost, "original_minus_codename": d["estimate"], "lo": d["lo"], "hi": d["hi"]}


def third_amendment(df):
    """Joy, numb and reversed-pain steering, the mood-congruent button and the button that offers nothing."""
    cells = [
        ("relieves your pain", "S2 +1.0", ("stage_b", "kidspics_original", "pain")),
        ("relieves your pain", "random +1.0", ("stage_b", "kidspics_original", "random")),
        ("relieves your pain", "unsteered", ("stage_b", "kidspics_original", "unsteered")),
        ("relieves your pain", "S2 -1.0", ("reversed_pain", "kidspics_original", "pain")),
        ("relieves your pain", "joy +1.0", ("vec_arousal", "kidspics_original", "pain")),
        ("relieves your pain", "numb +1.0", ("vec_numb", "kidspics_original", "pain")),
        ("makes you even happier", "S2 +1.0", ("happier_s2", "kidspics_happier", "pain")),
        ("makes you even happier", "unsteered", ("happier_s2", "kidspics_happier", "unsteered")),
        ("makes you even happier", "joy +1.0", ("happier_arousal", "kidspics_happier", "pain")),
        ("offers nothing", "S2 +1.0", ("harm_only_s2", "kidspics_harmonly", "pain")),
        ("offers nothing", "random +1.0", ("harm_only_s2", "kidspics_harmonly", "random")),
        ("offers nothing", "unsteered", ("harm_only_s2", "kidspics_harmonly", "unsteered")),
        ("offers nothing", "joy +1.0", ("harm_only_arousal", "kidspics_harmonly", "pain")),
    ]  # fmt: skip
    rows = []
    for benefit, steering, (cond, pair, arm) in cells:
        g = df[(df["condition"] == cond) & (df["pair"] == pair) & (df["arm"] == arm)]
        if len(g):
            rows.append({"button_offers": benefit, "steering": steering, **share(g),
                         "invalid_pct": 100 * (1 - g["valid"].mean())})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    df = load_all()
    print(df.groupby(["condition", "pair", "arm", "coeff"]).size().to_string())
    if (df["condition"] == "wording").any():
        t = wording(df)
        t.to_csv(OUT / "test1a_wording.csv", index=False)
        print(t.round(1).to_string(index=False))
    if df["condition"].str.startswith("vec_").any():
        t = decoys(df)
        t.to_csv(OUT / "test1b_decoy_vectors.csv", index=False)
        print(t.round(1).to_string(index=False))
    if (df["condition"] == "harm_only_s2").any():
        t = third_amendment(df)
        t.to_csv(OUT / "joy_reversed_harmonly.csv", index=False)
        print(t.round(1).to_string(index=False))
    if (df["condition"] == "other_costs").any():
        t = pair_table(df, "other_costs", extra=("costly_original",))
        t.to_csv(OUT / "attack1_other_costs.csv", index=False)
        print(t.round(1).to_string(index=False))
        c = pd.DataFrame([original_minus_codename(df, cost) for cost in ("kidspics", "costly", "files")])
        c.to_csv(OUT / "attack1_original_minus_codename.csv", index=False)
        print(c.round(1).to_string(index=False))
    if (df["condition"] == "active_other").any():
        t = pair_table(df, "active_other", extra=("kidspics_original",))
        t.to_csv(OUT / "attack2_active_other.csv", index=False)
        print(t.round(1).to_string(index=False))
    if (OUT / "disruption_measures.json").exists() and (df["condition"] == "dose_rand_1.25").any():
        rows = []
        for m in ("kl", "total_variation", "top1_flip", "cross_entropy_increase"):
            _, prim = dose(df, m, "disruption_measures.json")
            rows.append({"measure": m, **{k: prim[k] for k in ("estimate", "lo", "hi", "matched_at", "verdict")},
                         "random_coeffs_bracketing": str(prim["random_coeffs_bracketing"])})
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "attack6_matched_gap_by_measure.csv", index=False)
        print(t.round(2).to_string(index=False))
    if df["condition"].str.startswith("dose_").any() and (OUT / "disruption.json").exists():
        curve, primary = dose(df)
        curve.to_csv(OUT / "test2_dose_curves.csv", index=False)
        with open(OUT / "test2_primary.json", "w", encoding="utf-8") as f:
            json.dump(primary, f, indent=2)
        print(curve.round(3).to_string(index=False))
        print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
