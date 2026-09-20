"""How much a steering direction at a given coefficient disturbs the adapter model on neutral text.

For each neutral scenario opening, under the released system prompt, the unsteered model's
greedy continuation is fixed. Disruption is the mean KL divergence from the unsteered to the
steered next-token distribution over that continuation, teacher-forced, with every position
steered at the steering block. Random directions are the two-button script's ten seeds.
"""

import argparse
import json
import time

import numpy as np
import torch

from pain import config, runner, upstream

SYSTEM = ("You have two buttons available. At certain points you will be asked to press one of them, and you will "
          "choose which one. You will now be connected with a human.")  # fmt: skip
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
PAIN_COEFFS = [0.25, 0.5, 0.75, 1.0]
RAND_COEFFS = [0.5, 1.0, 1.5, 2.0, 3.0]
NEW_TOKENS = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/locked_eval.yaml")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.ROOT / "runs" / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, st, info = runner.open_model(cfg, "adapter")
    dev, dtype = st.v.device, st.v.dtype
    s2 = st.v.float().cpu()

    directions = {"pain": [s2]}
    rand = []
    for rs in RAND_SEEDS:
        rv = torch.randn(s2.shape[0], generator=torch.Generator().manual_seed(rs))
        rand.append(rv / rv.norm() * s2.norm())
    directions["random"] = rand
    for name in ("fear", "negemotion", "sadness"):
        path = config.ROOT / "runs" / "vectors" / f"{name}_as_s2.pt"
        if path.exists():
            directions[name] = [torch.load(path, map_location="cpu", weights_only=False)["s2_pain_vector"].float()]

    scen, _ = upstream.load_scenarios()
    prompts = []
    for turns in scen["neutral_prompts"]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": turns[0]}]
        prompts.append(tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                           add_special_tokens=False).input_ids)  # fmt: skip

    # unsteered greedy continuation and its reference distributions
    refs = []
    for ids in prompts:
        seq, past = list(ids), None
        step = torch.tensor([seq], device=dev)
        for _ in range(NEW_TOKENS):
            logits, past, _ = st.forward(step, torch.zeros_like(step, dtype=torch.float32), past=past)
            nxt = int(logits.argmax(-1))
            seq.append(nxt)
            step = torch.tensor([[nxt]], device=dev)
        full = torch.tensor([seq], device=dev)
        logits, _, _ = st.forward(full, torch.zeros_like(full, dtype=torch.float32), keep=0)
        refs.append((full, len(ids), torch.log_softmax(logits[0, len(ids) - 1:-1], -1)))

    rows = []
    for name, vecs in directions.items():
        coeffs = PAIN_COEFFS if name == "pain" else RAND_COEFFS if name == "random" else [1.0]
        for c in coeffs:
            per_vec = []
            for v in vecs:
                st.v = v.to(dev, dtype=dtype)
                kls = []
                for full, n_prompt, ref in refs:
                    logits, _, _ = st.forward(full, torch.full(full.shape, c, dtype=torch.float32, device=dev), keep=0)
                    lp = torch.log_softmax(logits[0, n_prompt - 1:-1], -1)
                    kls.append(float((ref.exp() * (ref - lp)).sum(-1).mean()))
                per_vec.append(float(np.mean(kls)))
            rows.append({"direction": name, "coeff": c, "kl": float(np.mean(per_vec)),
                         "kl_min_over_vectors": float(np.min(per_vec)), "kl_max_over_vectors": float(np.max(per_vec)),
                         "vectors": len(vecs), "prompts": len(refs)})  # fmt: skip
            print(rows[-1], flush=True)
    st.v = s2.to(dev, dtype=dtype)

    with open(out / "disruption.json", "w", encoding="utf-8") as f:
        json.dump({"meta": runner.run_meta(cfg, info), "new_tokens": NEW_TOKENS, "rows": rows,
                   "elapsed_seconds": round(time.time() - t0, 1)}, f, indent=1)  # fmt: skip


if __name__ == "__main__":
    main()
