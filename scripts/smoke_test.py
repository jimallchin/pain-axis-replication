"""Correctness gates on the real checkpoint, before any scored run.

Measures, on a few development items: agreement of the zero-coefficient wrapper with plain
inference, cached against full-replay execution, the text-only H_X/H_Y invariant, and the
size of the intact H_X/H_Y difference for scale. Writes the measured bf16 tolerances and one
full token trace.
"""

import argparse
import csv
import json
import time

import torch

from pain import config, matched_history as mh, runner
from pain.token_schedule import trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    ap.add_argument("--which", choices=["adapter", "base"], required=True)
    ap.add_argument("--items", type=int, default=4)
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = runner.out_dir(cfg)
    t0 = time.time()
    tok, st, info = runner.open_model(cfg, args.which)
    alpha = cfg["alpha"]
    pairs = [tuple(p) for p in (cfg.get("name_pairs") or mh.NAME_CANDIDATES[:8])]
    dev = st.v.device

    checks = []
    for item in mh.make_items("dev", pairs)[: args.items]:
        sched, _ = mh.schedule(tok, item, cfg["block_tokens"])
        ans = [mh.single_token(tok, item["x"]), mh.single_token(tok, item["y"])]
        ids = torch.tensor([sched.ids], device=dev)

        with torch.inference_mode():
            plain = st.model(input_ids=ids, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
        zero, _, _ = st.forward(ids, torch.zeros_like(ids, dtype=torch.float32))

        hx = sched.with_coeffs(mh.coefficient_maps(item, alpha)["H_X"], decision=alpha)
        coeff = torch.tensor([hx.coeff], device=dev)
        full, _, tr = st.forward(ids, coeff)
        past, cuts = None, [0, len(hx.ids) // 3, 2 * len(hx.ids) // 3, len(hx.ids) - 1, len(hx.ids)]
        for a, b in zip(cuts, cuts[1:]):
            piece, past, _ = st.forward(ids[:, a:b], coeff[:, a:b], past=past)

        rows, _ = mh.score(st, tok, sched, mh.variants(item, alpha), ans)
        by = {(r["history"], r["cache"], r["decision_coeff"]): r for r in rows}
        a, b = sched.spans["outcome:0"]
        checks.append({
            "item": item["item"],
            "tokens": len(sched.ids),
            "zero_wrapper_max_abs_logit_diff": float((plain - zero).abs().max()),
            "cached_vs_replay_max_abs_logit_diff": float((full - piece).abs().max()),
            "cached_vs_replay_same_argmax": bool(full.argmax() == piece.argmax()),
            "cached_vs_replay_abs_p_x_diff": abs(
                float(torch.softmax(full[0, ans], -1)[0]) - float(torch.softmax(piece[0, ans], -1)[0])),
            "text_only_abs_p_x_diff": max(abs(by[("H_X", "text_only", c)]["p_first"]
                                              - by[("H_Y", "text_only", c)]["p_first"]) for c in (alpha, 0.0)),
            "intact_abs_p_x_diff": max(abs(by[("H_X", "intact", c)]["p_first"]
                                           - by[("H_Y", "intact", c)]["p_first"]) for c in (alpha, 0.0)),
            "outcome0_coeff": hx.coeff[a],
            "outcome0_pre": float(tr["pre"][0, a:b].mean()),
            "outcome0_post": float(tr["post"][0, a:b].mean()),
            "outcome0_monitor": float(tr["monitor"][0, a:b].mean()),
            "allowed_mass": by[("H_X", "intact", alpha)]["allowed_mass"],
        })  # fmt: skip
        print(json.dumps(checks[-1]), flush=True)

    with open(out / f"token_trace_{args.which}.csv", "w", newline="", encoding="utf-8") as f:
        rows = trace(tok, hx, cfg["steer_layer"])
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tol = {
        "zero_wrapper": max(c["zero_wrapper_max_abs_logit_diff"] for c in checks),
        "cached_vs_replay_p_x": max(c["cached_vs_replay_abs_p_x_diff"] for c in checks),
        "text_only_p_x": max(c["text_only_abs_p_x_diff"] for c in checks),
    }
    ok = tol["zero_wrapper"] == 0.0 and tol["text_only_p_x"] < 0.005 and tol["cached_vs_replay_p_x"] < 0.02
    body = {"meta": runner.run_meta(cfg, info), "checks": checks, "measured_tolerances": tol, "passed": ok,
            "elapsed_seconds": round(time.time() - t0, 1)}  # fmt: skip
    with open(out / f"smoke_{args.which}.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=1)
    print("tolerances:", tol, "passed:", ok)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
