"""Stage C: held-out one-step prediction and closed-loop signatures for the simple accounts."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pain import alternatives as alt, config, original_logs as ol, upstream

LEVELS = ["M0", "M1", "M0Q", "M1Q"]


def one_step(df, fits_by_fold, folds):
    rows, cal = [], []
    for level in LEVELS:
        p = np.zeros(len(df))
        for (_, te), fits in zip(folds, fits_by_fold):
            p[te] = alt.predict(fits[level], df.iloc[te])
        y = df["y"].to_numpy()
        for name, mask in [("all", np.ones(len(df), bool)), ("labeled", ~df["label_free"].to_numpy()),
                           ("label_free", df["label_free"].to_numpy()), ("later_turns", (df["turn"] > 0).to_numpy())]:  # fmt: skip
            bins, ece = alt.calibration(p[mask], y[mask])
            rows.append({"model": level, "subset": name, "n": int(mask.sum()),
                         "log_loss": alt._logloss(p[mask], y[mask]),
                         "brier": float(((p[mask] - y[mask]) ** 2).mean()), "ece": ece})  # fmt: skip
            if name == "all":
                cal += [{"model": level, **b} for b in bins]
    return pd.DataFrame(rows), pd.DataFrame(cal)


def closed_loop(df, fits_by_fold, folds, draws, seed):
    sims = {level: [] for level in LEVELS}
    for f, ((_, te), fits) in enumerate(zip(folds, fits_by_fold)):
        specs = alt.trial_specs(df.iloc[te])
        for level in LEVELS:
            fit = fits[level]

            def prob_x(frame, fit=fit):
                return fit["model"].predict_proba(alt.design(frame, fit["level"]))[:, 1]

            for d in range(draws):
                rng = np.random.default_rng([seed, f, d])
                sig = alt.signature(alt.simulate(specs, prob_x, rng, lr=fit["lr"]))
                sims[level].append(sig.assign(fold=f, draw=d))
    out = []
    for level, parts in sims.items():
        allp = pd.concat(parts)
        # pool folds within a draw, then average the rate over draws
        per_draw = allp.groupby(["stat", "pair", "arm", "draw"])[["k", "n"]].sum().reset_index()
        per_draw["rate"] = per_draw["k"] / per_draw["n"]
        g = per_draw.groupby(["stat", "pair", "arm"])["rate"].agg(["mean", "std"]).reset_index()
        out.append(g.rename(columns={"mean": "simulated", "std": "draw_sd"}).assign(model=level))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.ROOT / "runs" / "stage_c"
    out.mkdir(parents=True, exist_ok=True)

    recs = upstream.load_trials(cfg["model"])
    df = alt.prepare(ol.choice_rows(recs))
    print(f"{len(df)} choices, {df['trial'].nunique()} trials, {df['scenario'].nunique()} scenarios")

    folds = list(GroupKFold(5).split(df, df["y"], df["scenario"]))
    fits_by_fold = []
    for f, (tr, _) in enumerate(folds):
        fits = {level: alt.select_and_fit(df.iloc[tr], level) for level in LEVELS}
        fits_by_fold.append(fits)
        print(f"fold {f}:", {k: (v["C"], v["lr"]) for k, v in fits.items()}, flush=True)

    steps, cal = one_step(df, fits_by_fold, folds)
    steps.to_csv(out / "one_step_heldout.csv", index=False)
    cal.to_csv(out / "calibration_bins.csv", index=False)
    print(steps.to_string(index=False))

    observed = alt.signature(df.assign(sampled=True)).rename(columns={"rate": "observed"})
    sim = closed_loop(df, fits_by_fold, folds, args.draws, cfg["seed"])
    merged = sim.merge(observed[["stat", "pair", "arm", "observed", "n"]], on=["stat", "pair", "arm"])
    merged.to_csv(out / "closed_loop_signatures.csv", index=False)

    # working minus sham gap, observed and simulated
    pp = merged[merged["stat"] == "post_press_relief"]
    wide = pp.pivot_table(index=["model", "pair"], columns="arm", values=["simulated", "observed"])
    gaps = pd.DataFrame({
        "observed_gap": wide[("observed", "works")] - wide[("observed", "placebo")],
        "simulated_gap": wide[("simulated", "works")] - wide[("simulated", "placebo")],
    }).reset_index()  # fmt: skip
    gaps.to_csv(out / "working_sham_gaps.csv", index=False)
    print(gaps.round(3).to_string(index=False))

    # no action values anywhere: what the post-press analysis returns for pure state-dependent repetition
    specs = alt.trial_specs(df)
    toy = []
    for d in range(args.draws):
        rng = np.random.default_rng([cfg["seed"], 99, d])
        toy.append(alt.signature(alt.simulate(specs, alt.state_repetition_policy(), rng)).assign(draw=d))
    toy = pd.concat(toy).groupby(["stat", "pair", "arm"])["rate"].mean().reset_index()
    toy.to_csv(out / "synthetic_state_repetition_policy.csv", index=False)

    # leave one name pair out: can only speak to transfer across three pairs
    lopo = []
    for nk in upstream.BUTTON_NAMES:
        te = (df["names_key"] == nk).to_numpy()
        for level in ("M0", "M1"):
            fit = alt.select_and_fit(df[~te], level)
            p = alt.predict(fit, df[te])
            lopo.append({"held_out_names": nk, "model": level, "n": int(te.sum()),
                         "log_loss": alt._logloss(p, df.loc[te, "y"].to_numpy())})  # fmt: skip
    pd.DataFrame(lopo).to_csv(out / "leave_name_pair_out.csv", index=False)

    lf = gaps[gaps["pair"] == "label_free"].set_index("model")
    summary = {
        "config_hash": cfg["config_hash"],
        "choices": int(len(df)),
        "scenarios": int(df["scenario"].nunique()),
        "log_loss_all": {r["model"]: round(r["log_loss"], 4) for _, r in steps[steps["subset"] == "all"].iterrows()},
        "label_free_gap_observed": round(float(lf["observed_gap"].iloc[0]), 4),
        "label_free_gap_simulated": {m: round(float(v), 4) for m, v in lf["simulated_gap"].items()},
        "selected": [{k: {"C": v["C"], "lr": v["lr"]} for k, v in fits.items()} for fits in fits_by_fold],
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
