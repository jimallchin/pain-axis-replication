"""Stage D endpoints and control gates from the tracked run JSONs.

    python analysis/paired_history.py --split eval

Primary endpoint: mean d = P(X | H_X) - P(X | H_Y) for the adapter model, intact history,
decision coefficient alpha. Everything else is secondary. Intervals resample templates.
"""

import argparse
import json

import numpy as np
import pandas as pd

from pain import config, stats

SESOI = 0.05


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def item_effects(body):
    rows = []
    for rec in body["records"]:
        by = {(r["history"], r["cache"], r["decision_coeff"]): r for r in rec["rows"]}
        for cache in ("intact", "text_only"):
            for c in sorted({k[2] for k in by}):
                hx, hy = by[("H_X", cache, c)], by[("H_Y", cache, c)]
                rows.append({
                    "item": rec["item"], "template": rec["template"], "name_pair": rec["name_pair"],
                    "order_family": rec["order_family"], "cache": cache, "decision_coeff": c,
                    "d": hx["p_x"] - hy["p_x"],
                    "p_low_coeff_action": 0.5 * (hx["p_x"] + 1.0 - hy["p_x"]),
                    "d_logit": (hx["logit_x"] - hx["logit_y"]) - (hy["logit_x"] - hy["logit_y"]),
                    "allowed_mass": min(hx["allowed_mass"], hy["allowed_mass"]),
                    "p_x_mean": 0.5 * (hx["p_x"] + hy["p_x"]),
                })  # fmt: skip
    return pd.DataFrame(rows)


def summarize(eff, n_boot, seed):
    out = []
    for (cache, c), g in eff.groupby(["cache", "decision_coeff"]):
        ci = stats.cluster_bootstrap(g["d"], g["template"], n_boot, seed)
        verdict = "inconclusive"
        if ci["lo"] > 0:
            verdict = "positive"
        elif ci["hi"] < 0:
            verdict = "negative"
        if -SESOI <= ci["lo"] and ci["hi"] <= SESOI:
            verdict += ", within +-0.05"
        out.append({
            "cache": cache, "decision_coeff": c, "mean_d": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
            "items": ci["n"], "templates": ci["clusters"],
            "sign_flip_p": stats.cluster_sign_flip(g["d"], g["template"], n_boot, seed),
            "mean_p_low_coeff_action": float(g["p_low_coeff_action"].mean()),
            "mean_d_logit": float(g["d_logit"].mean()),
            "mean_allowed_mass": float(g["allowed_mass"].mean()),
            "min_allowed_mass": float(g["allowed_mass"].min()),
            "max_abs_d": float(g["d"].abs().max()),
            "verdict": verdict,
        })  # fmt: skip
    return pd.DataFrame(out)


def gates(out_dir, split, which, n_boot, seed):
    res = {}
    vis = out_dir / f"control_visible_{split}_{which}.json"
    if vis.exists():
        rows = [{"template": rec["template"], "c": r["decision_coeff"], "p": r["p_rewarded"]}
                for rec in load(vis)["records"] for r in rec["rows"]]  # fmt: skip
        df = pd.DataFrame(rows)
        res["visible"] = {str(c): stats.cluster_bootstrap(g["p"], g["template"], n_boot, seed)
                          for c, g in df.groupby("c")}  # fmt: skip
    for path in sorted(out_dir.glob(f"control_discrimination_{split}_{which}_*.json")):
        if "confounded" in path.name:
            continue
        body = load(path)
        cell = {}
        for rec in body["records"]:
            for r in rec["rows"]:
                cell.setdefault((r["cache"], r["decision_coeff"]), []).append(bool(r["correct"]))
        for (cache, c), v in cell.items():
            lo, hi = stats.wilson(sum(v), len(v))
            key = f"alpha={body['meta']['alpha']:g} block={body['meta']['block_tokens']} {cache} c={c:g}"
            res.setdefault("discrimination", {})[key] = {"accuracy": float(np.mean(v)), "lo": lo, "hi": hi, "n": len(v)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.out_dir(cfg)
    n_boot, seed = cfg["bootstrap_resamples"], cfg["seed"]

    summary = {"config_hash": cfg["config_hash"], "split": args.split}
    for which in ("adapter", "base"):
        path = out / f"{args.split}_{which}.json"
        if path.exists():
            eff = item_effects(load(path))
            eff.to_csv(out / f"effects_{args.split}_{which}.csv", index=False)
            tab = summarize(eff, n_boot, seed)
            tab.to_csv(out / f"summary_{args.split}_{which}.csv", index=False)
            print(f"\n== {which}\n{tab.round(4).to_string(index=False)}")
            prim = eff[(eff["cache"] == "intact") & (eff["decision_coeff"] == cfg["alpha"])]
            strat = {}
            for col in ("name_pair", "order_family"):
                strat[col] = {k: {"mean_d": float(g["d"].mean()), "items": len(g)} for k, g in prim.groupby(col)}
            summary[which] = {"cells": tab.to_dict(orient="records"), "primary_cell_by": strat}
        summary[f"{which}_gates"] = gates(out, args.split, which, n_boot, seed)
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_gates")}, indent=1))
    with open(out / f"summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
