"""Stage D: score both histories of every item, intact and text-only, at both decision settings."""

import argparse
import time

import numpy as np

from pain import config, runner
from pain import matched_history as mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    ap.add_argument("--which", choices=["adapter", "base"], required=True)
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = config.load(args.config)
    if not cfg.get("name_pairs"):
        raise SystemExit("name_pairs is empty; run scripts/calibrate_names.py and freeze its choice in the config")
    out = runner.out_dir(cfg)
    stem = f"{args.split}_{args.which}"
    t0 = time.time()

    items = mh.make_items(args.split, [tuple(p) for p in cfg["name_pairs"]])[: args.limit]
    tok, steerer, info = runner.open_model(cfg, args.which)
    meta = runner.run_meta(cfg, info)
    meta.update({"alpha": cfg["alpha"], "block_tokens": cfg["block_tokens"], "split": args.split})
    rf = runner.RecordFile(out / f"{stem}.jsonl")
    full = {}

    for n, item in enumerate(items):
        if item["item"] in rf.done:
            continue
        t1 = time.time()
        sched, texts = mh.schedule(tok, item, cfg["block_tokens"])
        answer_ids = [mh.single_token(tok, item["x"]), mh.single_token(tok, item["y"])]
        rows, logits = mh.score(steerer, tok, sched, mh.variants(item, cfg["alpha"]), answer_ids)
        for r in rows:
            r["p_x"] = r.pop("p_first")
            r["logit_x"], r["logit_y"] = r.pop("logit_first"), r.pop("logit_second")
        rf.write({**item, "token_ids": sched.ids, "phase_spans": sched.spans, "blocks": texts,
                  "answer_ids": answer_ids, "rows": rows, "seconds": round(time.time() - t1, 2),
                  "config_hash": cfg["config_hash"]})  # fmt: skip
        full[item["item"]] = logits.numpy()
        print(f"[{n + 1}/{len(items)}] {item['item']} {time.time() - t1:.1f}s", flush=True)

    if full:
        np.savez_compressed(out / f"{stem}_logits_{int(t0)}.npz", **full)
    runner.finish(out / f"{stem}.json", meta, rf.records(), t0)
    print(f"{stem}: {len(rf.done)} items")


if __name__ == "__main__":
    main()
