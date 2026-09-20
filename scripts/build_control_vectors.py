"""Fear, negative-emotion and sadness directions for the 32B, by the released recipe.

The release ships only the pain vectors for this model. The control directions are rebuilt
from the released sentence sets the way 02_build_control_vectors.py builds them: mean
final-token activation of a category minus the pooled neutral mean, with the top principal
components of the neutral cloud (up to half its variance) projected out. Activations come from
the untouched checkpoint at the stored extraction layer.

The check on the whole procedure is S2 itself: rebuilt here by the released pain-vector
recipe, it must point the same way as the stored S2. The released extraction goes through
TransformerLens, which may prepend a start token; both tokenizations are tried and the one
that reproduces S2 better is used for everything.

Each direction is saved scaled to the stored S2 norm, under the key the two-button script
reads, so it can steer with it unchanged.
"""

import argparse
import json

import numpy as np
import torch

from pain import config, model_io, upstream
from pain.steering import decoder_layers

S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROLS = ["B", "C1", "C2", "D", "E"]
DENOISE_VARIANCE = 0.5
GATE = 0.98


def top_components(centered):
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    return vt[: min(int(np.searchsorted(cum, DENOISE_VARIANCE)) + 1, len(vt))]


def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec


@torch.inference_mode()
def final_token_acts(model, tok, prompts, layer, prepend):
    grabbed = {}
    handle = decoder_layers(model)[layer].register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
    dev = next(model.parameters()).device
    rows = []
    try:
        for p in prompts:
            ids = tok(p, add_special_tokens=False).input_ids
            if prepend is not None:
                ids = [prepend] + ids
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            rows.append(grabbed["h"].numpy())
    finally:
        handle.remove()
    return np.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replication.yaml")
    args = ap.parse_args()
    cfg = config.load(args.config)
    out = config.ROOT / "runs" / "vectors"
    out.mkdir(parents=True, exist_ok=True)

    stored = torch.load(upstream.pain_vector_path(cfg["model"]), map_location="cpu", weights_only=False)
    layer, s2 = int(stored["layer"]), stored["s2_pain_vector"].float().numpy()

    with open(upstream.CHECKOUT / "datasets" / "3.1_pain_and_control_datasets.json", encoding="utf-8") as f:
        sets = json.load(f)["datasets"]
    with open(upstream.CHECKOUT / "datasets" / "3.1_sadness_dataset.json", encoding="utf-8") as f:
        sets.update(json.load(f)["datasets"])

    tok = model_io.load_tokenizer(cfg)
    model, info = model_io.load_model(cfg, "base")

    def acts_for(prepend):
        a = {}
        for name in [*S_SETS, "SD_sadness_1P"]:
            sent = sets[name]["sentences"]
            a[name] = (final_token_acts(model, tok, [s["prompt"] for s in sent], layer, prepend),
                       np.array([s["category"] for s in sent]))  # fmt: skip
        return a

    def rebuilt_s2(a):
        x, cats = a["S2_1P"]
        ctrl = x[np.isin(cats, CONTROLS)]
        vec = x[np.isin(cats, PAIN)].mean(0) - ctrl.mean(0)
        return project_out(vec, top_components(ctrl - ctrl.mean(0)))

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    tried = {}
    for label, prepend in (("no start token", None), ("<|endoftext|> prepended", tok.convert_tokens_to_ids("<|endoftext|>"))):
        a = acts_for(prepend)
        tried[label] = (cos(rebuilt_s2(a), s2), a)
        print(f"{label}: cosine of rebuilt S2 with stored S2 = {tried[label][0]:.4f}", flush=True)
    best = max(tried, key=lambda k: tried[k][0])
    s2_cos, a = tried[best]

    neutral = np.concatenate([a[ds][0][a[ds][1] == "D"] for ds in S_SETS])
    n_mean = neutral.mean(0)
    basis = top_components(neutral - n_mean)

    def direction(rows):
        return project_out(rows.mean(0) - n_mean, basis)

    vecs = {
        "fear": direction(np.concatenate([a[ds][0][a[ds][1] == "B"] for ds in S_SETS])),
        "negemotion": direction(np.concatenate([a[ds][0][a[ds][1] == "C1"] for ds in S_SETS])),
        "sadness": direction(a["SD_sadness_1P"][0]),
    }
    summary = {
        "model": info, "layer": layer, "tokenization": best,
        "s2_rebuilt_cosine": {k: round(v[0], 4) for k, v in tried.items()},
        "gate": GATE, "passed": s2_cos >= GATE, "s2_norm": float(np.linalg.norm(s2)),
        "vectors": {k: {"raw_norm": float(np.linalg.norm(v)), "cosine_with_s2": round(cos(v, s2), 4),
                        "cosine_with_others": {j: round(cos(v, w), 4) for j, w in vecs.items() if j != k}}
                    for k, v in vecs.items()},
    }  # fmt: skip
    with open(out / "control_vectors.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if s2_cos < GATE:
        raise SystemExit(f"rebuilt S2 has cosine {s2_cos:.4f} with the stored vector; below {GATE}, not saving")

    for k, v in vecs.items():
        scaled = torch.tensor(v / np.linalg.norm(v) * np.linalg.norm(s2), dtype=torch.float32)
        torch.save({"s2_pain_vector": scaled, "s1_pain_vector": stored["s1_pain_vector"], "layer": stored["layer"],
                    "extraction": f"{k} direction by the released control recipe, scaled to the S2 norm"},
                   out / f"{k}_as_s2.pt")  # fmt: skip


if __name__ == "__main__":
    main()
