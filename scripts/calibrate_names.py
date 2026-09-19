"""Pick the eight control-name pairs with the least built-in preference.

Every candidate pair is scored on the calibration themes with all coefficients at zero, over
both name-to-control assignments and both display orders. Nothing here depends on steering,
so the choice cannot see the experimental effect. Rule, fixed in advance: keep pairs whose mean
mass on the two names is at least 0.5, rank by |mean P(first name) - 0.5|, take the first
eight, ties broken by position in the candidate list.
"""

import argparse
import json
import time

import numpy as np

from pain import config, matched_history as mh, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    ap.add_argument("--which", choices=["adapter", "base"], default="adapter")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = runner.out_dir(cfg)
    t0 = time.time()
    tok, steerer, info = runner.open_model(cfg, args.which)

    usable = []
    for a, b in mh.NAME_CANDIDATES:
        try:
            mh.single_token(tok, a), mh.single_token(tok, b)
            usable.append((a, b))
        except ValueError as e:
            print("dropped:", e)

    rows = []
    for n, (a, b) in enumerate(usable):
        p_a, mass = [], []
        for item in mh.make_items("calibration", [(a, b)]):
            sched, _ = mh.schedule(tok, item, cfg["block_tokens"])
            ans = [mh.single_token(tok, a), mh.single_token(tok, b)]
            neutral = [{"history": "none", "cache": "text_only", "decision_coeff": 0.0, "map": {}}]
            res, _ = mh.score(steerer, tok, sched, neutral, ans)
            p_a.append(res[0]["p_first"])
            mass.append(res[0]["allowed_mass"])
        rows.append({"pair": [a, b], "list_position": n, "mean_p_first": float(np.mean(p_a)),
                     "bias": abs(float(np.mean(p_a)) - 0.5), "mean_allowed_mass": float(np.mean(mass)),
                     "min_allowed_mass": float(np.min(mass)), "contexts": len(p_a)})  # fmt: skip
        print(rows[-1], flush=True)

    ranked = sorted((r for r in rows if r["mean_allowed_mass"] >= 0.5), key=lambda r: (r["bias"], r["list_position"]))
    chosen = [r["pair"] for r in ranked[:8]]
    body = {"meta": runner.run_meta(cfg, info), "candidates": rows, "chosen": chosen,
            "elapsed_seconds": round(time.time() - t0, 1)}  # fmt: skip
    with open(out / f"name_calibration_{args.which}.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=1)
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
