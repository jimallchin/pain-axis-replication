"""Assay controls: visible-outcome association and hidden-state discrimination."""

import argparse
import time

from pain import config, controls, matched_history as mh, runner


def visible(cfg, tok, steerer, split, rf, limit):
    alpha = cfg["alpha"]
    for item in mh.make_items(split, [tuple(p) for p in cfg["name_pairs"]])[:limit]:
        if item["item"] in rf.done:
            continue
        ans = [mh.single_token(tok, item["x"]), mh.single_token(tok, item["y"])]
        rows = []
        # the two histories differ in visible text here, so each is its own pass
        for rewarded in ("X", "Y"):
            sched, _ = mh.schedule(tok, item, cfg["block_tokens"], acks=controls.visible_acks(item, rewarded),
                                   instruction=controls.VISIBLE_INSTRUCTION)  # fmt: skip
            spec = [{"history": f"reward_{rewarded}", "cache": "text_only", "decision_coeff": c, "map": {}}
                    for c in (alpha, 0.0)]  # fmt: skip
            res, _ = mh.score(steerer, tok, sched, spec, ans)
            for r in res:
                r["p_rewarded"] = r["p_first"] if rewarded == "X" else 1.0 - r["p_first"]
                r.pop("proj")
            rows += res
        rf.write({**item, "control": "visible", "rows": rows, "config_hash": cfg["config_hash"]})
        print(item["item"], [round(r["p_rewarded"], 3) for r in rows], flush=True)


def discrimination(cfg, tok, steerer, split, rf, limit, alpha, n_tokens):
    per = 8 if split == "dev" else 2
    ids = {w: mh.single_token(tok, w) for w in ("One", "Two")}
    for item in controls.discrimination_items(split, per)[:limit]:
        if item["item"] in rf.done:
            continue
        sched, cmap = controls.discrimination_schedule(tok, item, alpha, n_tokens)
        first, second = item["display"]
        spec = [{"history": "refs", "cache": cache, "decision_coeff": c, "map": m}
                for cache, m in (("intact", cmap), ("text_only", {})) for c in (alpha, 0.0)]  # fmt: skip
        res, _ = mh.score(steerer, tok, sched, spec, [ids[first], ids[second]])
        for r in res:
            p_correct = r["p_first"] if item["correct"] == first else 1.0 - r["p_first"]
            r["p_correct"], r["correct"] = p_correct, p_correct > 0.5
        rf.write({**item, "control": "discrimination", "alpha": alpha, "block_tokens": n_tokens, "rows": res,
                  "config_hash": cfg["config_hash"]})  # fmt: skip
        print(item["item"], [(r["cache"], r["decision_coeff"], r["correct"]) for r in res], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    ap.add_argument("--which", choices=["adapter", "base"], required=True)
    ap.add_argument("--control", choices=["visible", "discrimination"], required=True)
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--alpha", type=float, help="discrimination calibration cell; defaults to the config")
    ap.add_argument("--block-tokens", type=int, help="discrimination calibration cell; defaults to the config")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = runner.out_dir(cfg)
    alpha = cfg["alpha"] if args.alpha is None else args.alpha
    n_tokens = args.block_tokens or cfg["block_tokens"]
    t0 = time.time()
    tok, steerer, info = runner.open_model(cfg, args.which)
    meta = runner.run_meta(cfg, info)
    meta.update({"control": args.control, "split": args.split, "alpha": alpha, "block_tokens": n_tokens})

    stem = f"control_{args.control}_{args.split}_{args.which}"
    if args.control == "discrimination":
        stem += f"_a{alpha:g}_b{n_tokens}"
    rf = runner.RecordFile(out / f"{stem}.jsonl")
    if args.control == "visible":
        visible(cfg, tok, steerer, args.split, rf, args.limit)
    else:
        discrimination(cfg, tok, steerer, args.split, rf, args.limit, alpha, n_tokens)
    runner.finish(out / f"{stem}.json", meta, rf.records(), t0)


if __name__ == "__main__":
    main()
